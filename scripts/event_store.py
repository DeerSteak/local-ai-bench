"""Transactional append-only job events and rebuildable state projections."""

import hashlib
import json
import sqlite3
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from result_store import validate_json_data
from run_plan import RunPlan


EVENT_SCHEMA_VERSION = 1
ENTITY_TYPES = {"job", "stage", "case", "attempt", "sample"}
TERMINAL_STATES = {"complete", "invalid", "skipped", "timed_out", "interrupted", "failed"}
STATE_TRANSITIONS = {
    None: {"pending", "running"},
    "pending": {"running", *TERMINAL_STATES},
    "running": TERMINAL_STATES,
}
RECOVERABLE_STATES = {"failed", "interrupted", "invalid", "timed_out"}


def _canonical_json(value) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _event_digest(previous_digest: str, values: list) -> str:
    payload = _canonical_json([previous_digest, *values]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class JournalEvent:
    entity_type: str
    entity_id: str
    state: str
    payload: dict
    event_id: str = ""
    occurred_at: str = ""
    parent_id: str = ""

    def normalized(self):
        if self.entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown event entity type: {self.entity_type}")
        expected_prefix = f"{self.entity_type}_"
        if not self.entity_id.startswith(expected_prefix):
            raise ValueError(f"{self.entity_type} identity must start with {expected_prefix}")
        if self.state not in {"pending", "running", *TERMINAL_STATES, "recorded"}:
            raise ValueError(f"unknown event state: {self.state}")
        if self.entity_type == "sample" and self.state != "recorded":
            raise ValueError("sample events must use recorded state")
        if self.entity_type != "sample" and self.state == "recorded":
            raise ValueError("recorded state is reserved for samples")
        parent_prefix = {
            "job": "", "stage": "job_", "case": "stage_",
            "attempt": "case_", "sample": "attempt_",
        }[self.entity_type]
        if (parent_prefix and not self.parent_id.startswith(parent_prefix)) \
                or (not parent_prefix and self.parent_id):
            raise ValueError(f"invalid parent identity for {self.entity_type}")
        if not isinstance(self.payload, dict):
            raise ValueError("event payload must be an object")
        if self.entity_type == "sample":
            required = {"valid", "measurement", "errors"}
            if (not required.issubset(self.payload)
                    or not isinstance(self.payload["valid"], bool)
                    or not isinstance(self.payload["measurement"], dict)
                    or not isinstance(self.payload["errors"], list)
                    or self.payload["valid"] == bool(self.payload["errors"])):
                raise ValueError("sample payload requires consistent valid, measurement, and errors")
        validate_json_data(self.payload)
        return JournalEvent(
            self.entity_type, self.entity_id, self.state,
            json.loads(_canonical_json(self.payload)),
            self.event_id or str(uuid.uuid4()),
            self.occurred_at or datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            self.parent_id,
        )


def apply_event(projection: dict, event: JournalEvent) -> None:
    collection = "samples" if event.entity_type == "sample" else f"{event.entity_type}s"
    entities = projection[collection]
    parent_collection = {
        "stage": "jobs", "case": "stages", "attempt": "cases", "sample": "attempts",
    }.get(event.entity_type)
    if parent_collection and event.parent_id not in projection[parent_collection]:
        raise ValueError(f"missing parent for {event.entity_type}: {event.parent_id}")
    if event.entity_type == "sample":
        if event.entity_id in entities:
            raise ValueError(f"sample already recorded: {event.entity_id}")
        entities[event.entity_id] = {"parent_id": event.parent_id, **event.payload}
        return
    current = entities.get(event.entity_id)
    previous = current["state"] if current else None
    recovery_transition = (
        event.state == "running" and previous in RECOVERABLE_STATES
        and event.payload.get("recovery") in {"resume", "retry"}
        and event.entity_type in {"job", "stage", "case"}
    )
    if event.state not in STATE_TRANSITIONS.get(previous, set()) and not recovery_transition:
        raise ValueError(
            f"illegal {event.entity_type} transition for {event.entity_id}: "
            f"{previous} -> {event.state}"
        )
    if current and current["parent_id"] != event.parent_id:
        raise ValueError(f"parent identity changed for {event.entity_id}")
    retained = {key: value for key, value in (current or {}).items() if key != "state"}
    entities[event.entity_id] = {
        **retained, "state": event.state, "parent_id": event.parent_id, **event.payload,
    }


class EventStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self):
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                plan_json TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                resume_identity_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                previous_digest TEXT NOT NULL,
                digest TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS events_no_update
            BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete
            BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS jobs_no_update
            BEFORE UPDATE ON jobs BEGIN SELECT RAISE(ABORT, 'jobs are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS jobs_no_delete
            BEFORE DELETE ON jobs BEGIN SELECT RAISE(ABORT, 'jobs are immutable'); END;
        """)
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(jobs)")}
        if "resume_identity_json" not in columns:
            self.connection.execute(
                "ALTER TABLE jobs ADD COLUMN resume_identity_json TEXT NOT NULL DEFAULT '{}'",
            )
        self.connection.commit()

    def close(self):
        self.connection.close()

    def create_job(self, plan: RunPlan, resume_identity: dict | None = None) -> str:
        plan_json = _canonical_json(plan.to_dict())
        resume_identity = resume_identity or {
            "plan_id": plan.plan_id, "artifacts": {}, "runtimes": {}, "methodology": {},
        }
        validate_json_data(resume_identity)
        resume_identity_json = _canonical_json(resume_identity)
        with self.connection:
            self.connection.execute(
                """INSERT INTO jobs(
                    job_id, plan_json, plan_id, schema_version, resume_identity_json
                ) VALUES (?, ?, ?, ?, ?)""",
                (plan.job_id, plan_json, plan.plan_id, EVENT_SCHEMA_VERSION, resume_identity_json),
            )
        return plan.job_id

    def has_job(self, job_id: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,),
        ).fetchone() is not None

    def start_stage(self, plan: RunPlan, stage: str, resume_identity: dict | None = None) -> None:
        events = []
        if self.has_job(plan.job_id):
            if self.load_plan(plan.job_id) != plan:
                raise ValueError("stage plan does not match the journal job")
            if resume_identity is not None and self.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("stage resume identity does not match the journal job")
        else:
            self.create_job(plan, resume_identity)
            events.append(JournalEvent("job", plan.job_id, "running", {}))
        events.append(JournalEvent(
            "stage", plan.stage_id(stage), "running", {"stage": stage},
            parent_id=plan.job_id,
        ))
        self.append(plan.job_id, events)

    def resume_identity(self, job_id: str) -> dict:
        row = self.connection.execute(
            "SELECT resume_identity_json FROM jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"unknown job: {job_id}")
        return json.loads(row["resume_identity_json"])

    def load_plan(self, job_id: str) -> RunPlan:
        row = self.connection.execute(
            "SELECT plan_json FROM jobs WHERE job_id = ?", (job_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"unknown job: {job_id}")
        return RunPlan.from_dict(json.loads(row["plan_json"]))

    def last_sequence(self, job_id: str) -> int:
        row = self.connection.execute(
            "SELECT MAX(sequence) AS sequence FROM events WHERE job_id = ?", (job_id,),
        ).fetchone()
        return int(row["sequence"] or 0)

    def append(self, job_id: str, events: list[JournalEvent]) -> None:
        normalized = [event.normalized() for event in events]
        projection = self.rebuild(job_id)
        for event in normalized:
            apply_event(projection, event)
        row = self.connection.execute(
            "SELECT digest FROM events WHERE job_id = ? ORDER BY sequence DESC LIMIT 1", (job_id,),
        ).fetchone()
        previous = row["digest"] if row else "0" * 64
        with self.connection:
            for event in normalized:
                payload_json = _canonical_json(event.payload)
                values = [
                    event.event_id, job_id, event.entity_type, event.entity_id,
                    event.state, payload_json, event.occurred_at, event.parent_id,
                ]
                digest = _event_digest(previous, values)
                self.connection.execute(
                    """INSERT INTO events(
                        event_id, job_id, entity_type, entity_id, state, payload_json,
                        occurred_at, parent_id, previous_digest, digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (*values, previous, digest),
                )
                previous = digest

    def events(self, job_id: str) -> list[JournalEvent]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE job_id = ? ORDER BY sequence", (job_id,),
        ).fetchall()
        return [JournalEvent(
            row["entity_type"], row["entity_id"], row["state"],
            json.loads(row["payload_json"]), row["event_id"], row["occurred_at"],
            row["parent_id"],
        ) for row in rows]

    def rebuild(self, job_id: str) -> dict:
        if not self.connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone():
            raise ValueError(f"unknown job: {job_id}")
        projection = {"jobs": {}, "stages": {}, "cases": {}, "attempts": {}, "samples": {}}
        for event in self.events(job_id):
            apply_event(projection, event)
        return projection

    def verify(self, job_id: str) -> None:
        previous = "0" * 64
        rows = self.connection.execute(
            "SELECT * FROM events WHERE job_id = ? ORDER BY sequence", (job_id,),
        ).fetchall()
        for row in rows:
            values = [
                row["event_id"], row["job_id"], row["entity_type"], row["entity_id"],
                row["state"], row["payload_json"], row["occurred_at"], row["parent_id"],
            ]
            if row["previous_digest"] != previous or row["digest"] != _event_digest(previous, values):
                raise ValueError(f"event digest chain is invalid at sequence {row['sequence']}")
            previous = row["digest"]
        self.rebuild(job_id)

    def case_aggregate(self, job_id: str, case_id: str) -> dict:
        projection = self.rebuild(job_id)
        if case_id not in projection["cases"]:
            raise ValueError(f"unknown case: {case_id}")
        attempt_ids = {
            identity for identity, attempt in projection["attempts"].items()
            if attempt["parent_id"] == case_id
        }
        samples = [
            sample for sample in projection["samples"].values()
            if sample["parent_id"] in attempt_ids
        ]
        valid = [sample for sample in samples if sample["valid"]]
        numeric_keys = set.intersection(*(
            {key for key, value in sample["measurement"].items()
             if isinstance(value, (int, float)) and not isinstance(value, bool)}
            for sample in valid
        )) if valid else set()
        return {
            "attempts": len(attempt_ids), "samples": len(samples),
            "valid_samples": len(valid), "invalid_samples": len(samples) - len(valid),
            "measurement_means": {
                key: statistics.mean(sample["measurement"][key] for sample in valid)
                for key in sorted(numeric_keys)
            },
        }

    def prepare_recovery(self, job_id: str, stage: str,
                         selected_case_ids: list[str] | None = None) -> dict[str, int]:
        """Reopen recoverable state and return the next attempt number for selected cases."""
        plan = self.load_plan(job_id)
        if stage not in plan.stage_order:
            raise ValueError(f"stage is absent from run plan: {stage}")
        projection = self.rebuild(job_id)
        stage_id = plan.stage_id(stage)
        stage_state = projection["stages"].get(stage_id, {}).get("state")
        job_state = projection["jobs"].get(job_id, {}).get("state")
        if stage_state in {"complete", "skipped"} or job_state == "complete":
            raise ValueError("completed state cannot be resumed; create a fork")
        candidates = {
            case_id for case_id, case in projection["cases"].items()
            if case["parent_id"] == stage_id and case["state"] not in {"complete", "skipped"}
        }
        selected = candidates if selected_case_ids is None else set(selected_case_ids)
        unknown = selected - candidates
        if unknown:
            raise ValueError(f"cases are not retry-eligible: {', '.join(sorted(unknown))}")
        events = []
        for attempt_id, attempt in projection["attempts"].items():
            if attempt["state"] == "running" and attempt["parent_id"] in selected:
                events.append(JournalEvent(
                    "attempt", attempt_id, "interrupted", {"reason": "recovery"},
                    parent_id=attempt["parent_id"],
                ))
        for case_id in sorted(selected):
            case = projection["cases"][case_id]
            if case["state"] == "running":
                events.append(JournalEvent(
                    "case", case_id, "interrupted", {"reason": "recovery"},
                    parent_id=stage_id,
                ))
            events.append(JournalEvent(
                "case", case_id, "running", {"recovery": "retry"}, parent_id=stage_id,
            ))
        if stage_state in RECOVERABLE_STATES:
            events.append(JournalEvent(
                "stage", stage_id, "running", {"recovery": "resume"}, parent_id=job_id,
            ))
        if job_state in RECOVERABLE_STATES:
            events.append(JournalEvent("job", job_id, "running", {"recovery": "resume"}))
        if events:
            self.append(job_id, events)
        attempts = projection["attempts"]
        return {
            case_id: max((
                attempt.get("number", 0) for attempt in attempts.values()
                if attempt["parent_id"] == case_id
            ), default=0) + 1
            for case_id in sorted(selected)
        }
