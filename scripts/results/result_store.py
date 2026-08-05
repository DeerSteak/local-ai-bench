"""Durable JSON persistence and additive run-state metadata."""

import json
import math
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 4
TERMINAL_STATUSES = {"complete", "partial", "interrupted", "failed"}
NON_MEASUREMENT_KEYS = {
    "label", "skipped", "skip_reason", "skip_detail", "error", "crashed",
    "crashed_at", "timed_out", "timed_out_at", "slow_tps", "stopped_at",
    "partial", "reason", "requested_cases", "completed_cases",
    "requested_repetitions", "completed_repetitions",
    "memory_at_failure",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nonfinite_path(value, path="$") -> str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return path
    if isinstance(value, dict):
        for key, child in value.items():
            found = _nonfinite_path(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _nonfinite_path(child, f"{path}[{index}]")
            if found:
                return found
    return None


def atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    invalid_path = _nonfinite_path(data)
    if invalid_path:
        raise ValueError(f"non-finite numeric value at {invalid_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def validate_json_data(data) -> None:
    invalid_path = _nonfinite_path(data)
    if invalid_path:
        raise ValueError(f"non-finite numeric value at {invalid_path}")


def source_identity(repo_root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout != ""
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def model_identity(models: list[dict]) -> list[dict]:
    keys = ("tag", "short", "size_gb", "params_b")
    return [{key: model[key] for key in keys if key in model} for model in models]


def build_run_manifest(*, plan, repo_root: Path,
                       repetition_mode="streamed_internal_repetitions") -> dict:
    return {
        "run_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "started_at": utc_now(),
        "finished_at": None,
        "status": "running",
        "requested_tests": list(plan.tests),
        "stage_order": list(plan.stage_order),
        "engine": plan.engine_name,
        "models": plan.models,
        "effective_config": plan.effective_config,
        "plan_id": plan.plan_id,
        "job_id": plan.job_id,
        "plan": plan.to_dict(),
        "source": source_identity(repo_root),
        "llamabench_repetition_mode": repetition_mode,
        "stages": {},
    }


def start_stage(run: dict, stage: str, selected_models: int) -> None:
    run["stages"][stage] = {
        "status": "running", "started_at": utc_now(), "finished_at": None,
        "selected_models": selected_models, "models_with_results": 0,
        "models_skipped": 0, "models_failed": 0,
    }


def model_counts(section: dict) -> dict:
    values = [value for value in section.values() if isinstance(value, dict)]
    skipped = sum(bool(value.get("skipped")) for value in values)
    with_results = sum(
        _has_measurement_payload(value) for value in values if not value.get("skipped")
    )
    failed = len(values) - skipped - with_results
    return {
        "models_with_results": with_results,
        "models_skipped": skipped,
        "models_failed": failed,
    }


def _has_measurement_payload(value, key=None) -> bool:
    if key in NON_MEASUREMENT_KEYS or (isinstance(key, str) and key.endswith(("_ids", "_count"))):
        return False
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_measurement_payload(child, child_key) for child_key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_measurement_payload(child) for child in value)
    return False


def finish_active_stage(run: dict, status: str, reason: str) -> None:
    active = next((record for record in run.get("stages", {}).values()
                   if record.get("status") == "running"), None)
    if active is not None:
        active.update(status=status, finished_at=utc_now(), reason=reason)


def finish_stage(run: dict, stage: str, section: dict, status="complete", reason=None) -> None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal stage status: {status}")
    record = run["stages"][stage]
    counts = model_counts(section)
    accounted = sum(counts.values())
    counts["models_failed"] += max(0, record["selected_models"] - accounted)
    record.update(status=status, finished_at=utc_now(), **counts)
    if reason:
        record["reason"] = reason


def finish_run(run: dict, status: str, reason=None) -> None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal run status: {status}")
    run.update(status=status, finished_at=utc_now())
    if reason:
        run["reason"] = reason


class ResultStore:
    """Owns result mutation, state transitions, validation, and checkpoints."""

    def __init__(self, path: Path, data: dict, writer=atomic_write_json):
        self.path = Path(path)
        self.data = data
        self._writer = writer

    def checkpoint(self) -> None:
        validate_json_data(self.data)
        self._writer(self.path, self.data)

    def begin_recovery(self) -> None:
        run = self.data["run"]
        if run["status"] not in TERMINAL_STATUSES:
            raise ValueError("only a terminal run can begin recovery")
        history = run.setdefault("recovery_history", [])
        history.append({
            "status": run["status"], "finished_at": run.get("finished_at"),
            "reason": run.get("reason"),
        })
        run.update(status="running", finished_at=None)
        run.pop("reason", None)
        self.checkpoint()

    def resume_stage(self, key: str, selected_models: int) -> None:
        if self.data["run"]["status"] != "running":
            raise ValueError("cannot resume a stage outside a running recovery")
        record = self.data["run"]["stages"].get(key)
        if record is None:
            self.start_stage(key, selected_models)
            return
        if record["status"] not in TERMINAL_STATUSES:
            raise ValueError(f"stage is not terminal: {key}")
        history = record.setdefault("recovery_history", [])
        history.append({
            "status": record["status"], "finished_at": record.get("finished_at"),
            "reason": record.get("reason"),
        })
        record.update(status="running", finished_at=None, selected_models=selected_models)
        record.pop("reason", None)
        self.checkpoint()

    def start_stage(self, key: str, selected_models: int) -> None:
        if self.data["run"]["status"] != "running":
            raise ValueError("cannot start a stage after the run ended")
        existing = self.data["run"]["stages"].get(key)
        if existing and existing["status"] == "running":
            raise ValueError(f"stage already running: {key}")
        start_stage(self.data["run"], key, selected_models)
        self.checkpoint()

    def update_section(self, section: str, value: dict, stage: str | None = None) -> None:
        validate_json_data(value)
        self.data[section] = value
        record = self.data["run"]["stages"].get(stage or section)
        if record:
            record.update(model_counts(value))
        self.checkpoint()

    def complete_stage(self, key: str, section: str | None = None,
                       status="complete", reason=None) -> None:
        record = self.data["run"]["stages"].get(key)
        if not record or record["status"] != "running":
            raise ValueError(f"stage is not running: {key}")
        finish_stage(self.data["run"], key, self.data[section or key], status, reason)
        self.checkpoint()

    def record_cleanup_failure(self, key: str, exc: BaseException) -> None:
        record = self.data["run"]["stages"].get(key)
        if not record or record["status"] != "running":
            raise ValueError(f"stage is not running: {key}")
        record["cleanup_failure"] = {
            "reason": "stage_cleanup_failed", "error_type": type(exc).__name__,
        }

    def finish(self, status: str, reason=None) -> None:
        if self.data["run"]["status"] != "running":
            raise ValueError("run is already terminal")
        finish_run(self.data["run"], status, reason)
        self.checkpoint()
