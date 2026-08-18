"""Journal-owned image resolutions and compatible projection."""

import math
from pathlib import Path

from scripts.results.content_store import ArtifactRef
from scripts.results.accuracy_event_stage import merge_memory_evidence, merge_power_evidence
from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan
from scripts.runtime.shared import Shared


class ImageEventStage:
    def __init__(self, path: Path, plan: RunPlan, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 selected_case_ids: list[str] | None = None):
        self.plan = plan
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("img")
        self.store = EventStore(path)
        self.model_identities = {model.get("short"): model for model in plan.models["images"]}
        self.recovery_attempts = {}
        if resume:
            if not initialize:
                raise ValueError("resume requires an initializing stage owner")
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.recovery_attempts = self.store.prepare_recovery(
                plan.job_id, "img", selected_case_ids,
            )
        elif initialize:
            self.store.start_stage(plan, "img", resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self) -> None:
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["short"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['short']}")
        return self.plan.model_id("images", identity)

    def _case_id(self, model: dict, width: int, height: int) -> str:
        return self.plan.case_id(
            "img", self._model_id(model), {"width": width, "height": height},
        )

    def next_attempt(self, model: dict, width: int, height: int) -> int | None:
        case_id = self._case_id(model, width, height)
        projection = self.store.rebuild(self.plan.job_id)
        case = projection["cases"].get(case_id)
        if case is None:
            return 1
        if case["state"] in {"complete", "skipped"}:
            return None
        if case_id in self.recovery_attempts:
            return self.recovery_attempts[case_id]
        stage = projection["stages"].get(self.stage_id, {})
        if (stage.get("recovery_scope") == "selected"
                and case_id not in stage.get("selected_case_ids", [])):
            return None
        if case.get("recovery") == "retry":
            numbers = [
                attempt.get("number", 0) for attempt in projection["attempts"].values()
                if attempt["parent_id"] == case_id
            ]
            return max(numbers, default=0) + 1
        raise ValueError("incomplete image resolution was not prepared for recovery")

    def pending_resolutions(self, model: dict, resolutions) -> list[tuple[int, int, int]]:
        pending = []
        for width, height in resolutions:
            attempt = self.next_attempt(model, width, height)
            if attempt is not None:
                pending.append((width, height, attempt))
        return pending

    def record_resolution(self, model: dict, width: int, height: int, times: list[float],
                          requested_runs: int, status: str = "ok", *,
                          attempt_number: int = 1, artifact: dict | None = None,
                          failure_detail: str | None = None) -> None:
        if status not in {"ok", "failed", "timed_out"}:
            raise ValueError(f"unknown image resolution status: {status}")
        if (not isinstance(width, int) or isinstance(width, bool) or width < 1
                or not isinstance(height, int) or isinstance(height, bool) or height < 1):
            raise ValueError("image resolution dimensions must be positive integers")
        if not isinstance(requested_runs, int) or isinstance(requested_runs, bool) \
                or requested_runs < 1:
            raise ValueError("image resolution requires a positive run count")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError("image timings must be finite and positive")
        if artifact is not None:
            artifact = ArtifactRef.from_dict(artifact).to_dict()
        case_id = self._case_id(model, width, height)
        projection = self.store.rebuild(self.plan.job_id)
        existing = projection["cases"].get(case_id)
        if existing and existing["state"] in {"complete", "skipped"}:
            raise ValueError("image resolution already completed")
        if existing and existing["state"] != "running":
            raise ValueError("incomplete image resolution was not prepared for recovery")
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "image_resolution", "model_short": model["short"],
                "model_label": model["label"], "checkpoint": model["checkpoint"],
                "steps": model["steps"], "width": width, "height": height,
                "requested_runs": requested_runs,
            }, parent_id=self.stage_id))
        events.append(JournalEvent(
            "attempt", attempt_id, "running", {"number": attempt_number}, parent_id=case_id,
        ))
        for number, elapsed in enumerate(times, 1):
            events.append(JournalEvent(
                "sample", self.plan.sample_id(attempt_id, number), "recorded",
                {"number": number, "valid": True,
                 "measurement": {"elapsed_sec": elapsed}, "errors": []},
                parent_id=attempt_id,
            ))
        terminal = "complete" if status == "ok" else status
        events.extend([
            JournalEvent("attempt", attempt_id, terminal, {}, parent_id=case_id),
            JournalEvent("case", case_id, terminal, {
                "artifact": artifact, "failure_detail": failure_detail,
            }, parent_id=self.stage_id),
        ])
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        if state not in {"skipped", "failed"}:
            raise ValueError(f"invalid image model state: {state}")
        case_id = self.plan.case_id(
            "img", self._model_id(model), {"model_state": state},
        )
        existing = self.store.rebuild(self.plan.job_id)["cases"].get(case_id)
        if existing and existing["state"] != "running":
            return
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
                "model_label": model["label"],
            }, parent_id=self.stage_id))
        events.append(JournalEvent("case", case_id, state, {"model_result": result},
                                   parent_id=self.stage_id))
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def record_model_evidence(self, model: dict, memory, power=None) -> None:
        projection = self.store.rebuild(self.plan.job_id)
        count = sum(
            case.get("parent_id") == self.stage_id
            and case.get("case_kind") == "model_evidence"
            and case.get("model_short") == model["short"]
            for case in projection["cases"].values()
        )
        case_id = self.plan.case_id(
            "img", self._model_id(model), {"model_evidence": count + 1},
        )
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_evidence", "model_short": model["short"],
                "model_label": model["label"],
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, "complete", {"memory": memory, "power": power},
                         parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        results = {}
        evidence = {}
        model_states = {}
        cases = sorted(
            projection["cases"].items(),
            key=lambda item: (item[1].get("model_short", ""), item[1].get("width", 0),
                              item[1].get("height", 0), item[0]),
        )
        for case_id, case in cases:
            if case["parent_id"] != self.stage_id or case["state"] == "running":
                continue
            short = case["model_short"]
            if case["case_kind"] == "model_state":
                model_states.setdefault(short, []).append(case)
                continue
            if case["case_kind"] == "model_evidence":
                evidence.setdefault(short, []).append(case)
                continue
            result = results.setdefault(short, {
                "label": case["model_label"], "checkpoint": case["checkpoint"],
                "steps": case["steps"], "resolutions": {},
            })
            attempts = [
                (attempt_id, attempt) for attempt_id, attempt in projection["attempts"].items()
                if attempt["parent_id"] == case_id
            ]
            latest = max((attempt.get("number", 0) for _, attempt in attempts), default=0)
            attempt_ids = {attempt_id for attempt_id, attempt in attempts
                           if attempt.get("number") == latest}
            samples = sorted(
                (sample for sample in projection["samples"].values()
                 if sample["parent_id"] in attempt_ids),
                key=lambda sample: sample["number"],
            )
            times = [sample["measurement"]["elapsed_sec"] for sample in samples]
            if times:
                label = f"{case['width']}x{case['height']}"
                result["resolutions"][label] = {
                    "sec_per_image_mean": round(Shared.mean(times), 2),
                    "sec_per_image_stdev": round(Shared.stdev(times), 2)
                    if len(times) > 1 else 0.0,
                    "n_runs": len(times), "runs": [round(value, 2) for value in times],
                }
            if case["state"] == "timed_out":
                result["timed_out"] = f"{case['width']}x{case['height']}"
        for short, cases_for_model in evidence.items():
            if short not in results:
                continue
            memory = merge_memory_evidence([case.get("memory") for case in cases_for_model])
            power = merge_power_evidence([case.get("power") for case in cases_for_model])
            if memory is not None:
                results[short]["memory"] = memory
            if power is not None:
                results[short]["power"] = power
        for short, states in model_states.items():
            case = states[-1]
            results[short] = {"label": case["model_label"], **case["model_result"]}
        return results

    def finish(self) -> None:
        projection = self.store.rebuild(self.plan.job_id)
        unresolved = any(
            case["parent_id"] == self.stage_id
            and case["state"] not in {"complete", "skipped"}
            for case in projection["cases"].values()
        )
        stage = projection["stages"].get(self.stage_id, {})
        state = "failed" if stage.get("recovery_scope") == "selected" and unresolved \
            else "complete"
        self.store.append(self.plan.job_id, [
            JournalEvent("stage", self.stage_id, state, {}, parent_id=self.plan.job_id),
        ])
        self.export_fn(self.export())


def export_images(path: Path, job_id: str) -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = ImageEventStage(path, plan, lambda _: None, initialize=False)
    try:
        return stage.export()
    finally:
        stage.close()
