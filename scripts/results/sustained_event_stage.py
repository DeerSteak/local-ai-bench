"""Journal ownership and projection for sustained-load model soaks."""

from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.llm_event_stage import measurement_payload
from scripts.results.run_plan import RunPlan
from scripts.runtime.engines.base import GenerationMeasurement, measurement_validation_errors


class SustainedEventStage:
    def __init__(self, path: Path, plan: RunPlan, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 selected_case_ids: list[str] | None = None, telemetry=None):
        self.plan = plan
        self.store = EventStore(path)
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("sustained")
        self.model_identities = {model.get("tag"): model for model in plan.models["llm"]}
        self.telemetry = telemetry
        self.recovery_attempts = {}
        if resume:
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.recovery_attempts = self.store.prepare_recovery(
                plan.job_id, "sustained", selected_case_ids,
            )
        elif initialize:
            self.store.start_stage(plan, "sustained", resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self) -> None:
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("llm", identity)

    def _case_id(self, model: dict) -> str:
        return self.plan.case_id("sustained", self._model_id(model), {"soak": True})

    def next_attempt(self, model: dict) -> int | None:
        case_id = self._case_id(model)
        projection = self.store.rebuild(self.plan.job_id)
        case = projection["cases"].get(case_id)
        if case is None:
            return 1
        if case["state"] in {"complete", "skipped"}:
            return None
        if case_id in self.recovery_attempts:
            return self.recovery_attempts[case_id]
        raise ValueError("incomplete sustained case was not prepared for recovery")

    def begin_case(self, model: dict, attempt_number: int) -> None:
        case_id = self._case_id(model)
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        projection = self.store.rebuild(self.plan.job_id)
        events = []
        if case_id not in projection["cases"]:
            events.append(JournalEvent("case", case_id, "running", {
                "model_short": model["short"], "model_label": model["label"],
                "case_kind": "sustained",
            }, parent_id=self.stage_id))
        events.append(JournalEvent(
            "attempt", attempt_id, "running", {"number": attempt_number}, parent_id=case_id,
        ))
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def record_request(self, model: dict, attempt_number: int, number: int,
                       measurement: GenerationMeasurement, start_sec: float,
                       end_sec: float) -> None:
        case_id = self._case_id(model)
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        sample_id = self.plan.sample_id(attempt_id, number)
        errors = measurement_validation_errors(measurement)
        payload = measurement_payload(measurement)
        payload.update({"start_sec": start_sec, "end_sec": end_sec})
        self.store.append(self.plan.job_id, [JournalEvent(
            "sample", sample_id, "recorded", {
                "number": number, "valid": not errors, "measurement": payload,
                "errors": errors,
            }, parent_id=attempt_id,
        )])
        self.export_fn(self.export())

    def complete_case(self, model: dict, attempt_number: int, result: dict,
                      state: str = "complete") -> None:
        case_id = self._case_id(model)
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        self.store.append(self.plan.job_id, [
            JournalEvent("attempt", attempt_id, state, {}, parent_id=case_id),
            JournalEvent("case", case_id, state, {"result": result}, parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id("sustained", model_id, {"model_state": state})
        payload = {
                "model_short": model["short"], "model_label": model["label"],
                "case_kind": "model_state", "result": result,
        }
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", payload, parent_id=self.stage_id),
            JournalEvent("case", case_id, state, {}, parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def finish(self) -> None:
        projection = self.store.rebuild(self.plan.job_id)
        unresolved = any(
            case["parent_id"] == self.stage_id
            and case["state"] not in {"complete", "skipped"}
            for case in projection["cases"].values()
        )
        self.store.append(self.plan.job_id, [JournalEvent(
            "stage", self.stage_id, "failed" if unresolved else "complete", {},
            parent_id=self.plan.job_id,
        )])
        self.export_fn(self.export())

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        results = {}
        for case in projection["cases"].values():
            if case["parent_id"] != self.stage_id:
                continue
            short = case["model_short"]
            if case["case_kind"] == "model_state":
                results[short] = case.get("result", {})
            elif case["state"] != "running":
                results[short] = case.get("result", {})
        return results


def export_sustained_section(path: Path, job_id: str) -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = SustainedEventStage(path, plan, lambda _: None, initialize=False)
    try:
        return stage.export()
    finally:
        stage.close()
