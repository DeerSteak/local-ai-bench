"""Journal-owned single-shot LLM cases and compatible JSON projection."""

from dataclasses import asdict
from pathlib import Path

from scripts.runtime.engines.base import (
    GenerationMeasurement, aggregate_generation_measurements, measurement_validation_errors,
)
from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan


MEASUREMENT_FIELDS = {
    "client_ttft_sec", "generated_tokens", "tokens_per_sec", "client_wall_sec",
    "decode_sec", "server_prompt_sec", "prompt_tokens", "finish_reason",
    "model_load_sec", "server_tps_implausible",
}


def event_store_path(result_path: Path) -> Path:
    return Path(result_path).with_suffix(".events.sqlite3")


def measurement_payload(measurement: GenerationMeasurement) -> dict:
    values = asdict(measurement)
    return {key: values[key] for key in MEASUREMENT_FIELDS}


def measurement_from_payload(payload: dict) -> GenerationMeasurement:
    return GenerationMeasurement(**{
        key: payload[key] for key in MEASUREMENT_FIELDS if key in payload
    })


class LLMEventStage:
    def __init__(self, path: Path, plan: RunPlan, export_fn, *, stage_name: str = "llm",
                 model_family: str = "llm", initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 selected_case_ids: list[str] | None = None):
        self.plan = plan
        self.store = EventStore(path)
        self.export_fn = export_fn
        self.stage_name = stage_name
        self.model_family = model_family
        self.stage_id = plan.stage_id(stage_name)
        self.model_identities = {model.get("tag"): model for model in plan.models[model_family]}
        self.recovery_attempts = {}
        if resume:
            if not initialize:
                raise ValueError("resume requires an initializing stage owner")
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.recovery_attempts = self.store.prepare_recovery(
                plan.job_id, stage_name, selected_case_ids,
            )
        elif initialize:
            self.store.start_stage(plan, stage_name, resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self):
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id(self.model_family, identity)

    def next_context_attempt(self, model: dict, context_tokens: int) -> int | None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id(
            self.stage_name, model_id, {"context_tokens": context_tokens},
        )
        projection = self.store.rebuild(self.plan.job_id)
        case = projection["cases"].get(case_id)
        if case is None:
            return 1
        if case["state"] in {"complete", "skipped"}:
            return None
        if case_id in self.recovery_attempts:
            return self.recovery_attempts[case_id]
        if case.get("recovery") == "retry":
            numbers = [
                attempt.get("number", 0) for attempt in projection["attempts"].values()
                if attempt["parent_id"] == case_id
            ]
            return max(numbers, default=0) + 1
        stage = projection["stages"].get(self.stage_id, {})
        if stage.get("recovery_scope") == "selected":
            return None
        raise ValueError("incomplete case was not prepared for recovery")

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id(self.stage_name, model_id, {"model_state": state})
        projection = self.store.rebuild(self.plan.job_id)
        events = []
        if case_id not in projection["cases"]:
            events.append(JournalEvent("case", case_id, "running", {
                "model_short": model["short"], "model_label": model["label"],
                "case_kind": "model_state",
            }, parent_id=self.stage_id))
        elif projection["cases"][case_id]["state"] in {"complete", "skipped"}:
            self.export_fn(self.export())
            return
        elif projection["cases"][case_id]["state"] != "running":
            raise ValueError("model-state case was not prepared for recovery")
        events.append(
            JournalEvent("case", case_id, state, {"model_result": result},
                         parent_id=self.stage_id)
        )
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def record_case(self, model: dict, context_tokens: int, context_label: str,
                    samples: list[GenerationMeasurement], status: str,
                    requested_runs: int, model_markers: dict | None = None,
                    depth_tokens: int | None = None,
                    result_fields: dict | None = None, attempt_number: int = 1) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id(
            self.stage_name, model_id, {"context_tokens": context_tokens},
        )
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        projection = self.store.rebuild(self.plan.job_id)
        events = []
        if case_id not in projection["cases"]:
            events.append(JournalEvent("case", case_id, "running", {
                "model_short": model["short"], "model_label": model["label"],
                "case_kind": "context", "context_tokens": context_tokens,
                "context_label": context_label, "requested_runs": requested_runs,
                "depth_tokens": depth_tokens,
            }, parent_id=self.stage_id))
        events += [
            JournalEvent("attempt", attempt_id, "running", {"number": attempt_number},
                         parent_id=case_id),
        ]
        for number, sample in enumerate(samples, 1):
            errors = measurement_validation_errors(sample)
            sample_id = self.plan.sample_id(attempt_id, number)
            events.append(JournalEvent("sample", sample_id, "recorded", {
                "number": number, "valid": not errors,
                "measurement": measurement_payload(sample), "errors": errors,
            }, parent_id=attempt_id))
        case_state = {
            "timed_out": "timed_out", "crashed": "failed",
            "ok": "complete" if any(not measurement_validation_errors(sample)
                                      for sample in samples) else "invalid",
        }.get(status, "failed")
        attempt_state = "complete" if status == "ok" else case_state
        events.extend([
            JournalEvent("attempt", attempt_id, attempt_state, {}, parent_id=case_id),
            JournalEvent("case", case_id, case_state, {
                "run_status": status, "model_markers": model_markers or {},
                "result_fields": result_fields or {},
            }, parent_id=self.stage_id),
        ])
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def finish(self) -> None:
        models = self.export()
        with_results = sum(any(isinstance(value, dict) for value in model.values())
                           for model in models.values())
        projection = self.store.rebuild(self.plan.job_id)
        stage = projection["stages"].get(self.stage_id, {})
        unresolved = any(
            case["parent_id"] == self.stage_id
            and case["state"] not in {"complete", "skipped"}
            for case in projection["cases"].values()
        )
        state = "failed" if stage.get("recovery_scope") == "selected" and unresolved else "complete"
        self.store.append(self.plan.job_id, [
            JournalEvent("stage", self.stage_id, state, {
                "models_with_results": with_results,
            }, parent_id=self.plan.job_id),
        ])
        self.export_fn(models)

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        results = {}
        for case_id, case in projection["cases"].items():
            if case["parent_id"] != self.stage_id:
                continue
            short = case["model_short"]
            if case["state"] == "running":
                continue
            if case["case_kind"] == "model_state":
                results.setdefault(short, {}).update(case.get("model_result", {}))
                continue
            attempt_values = [
                (identity, attempt) for identity, attempt in projection["attempts"].items()
                if attempt["parent_id"] == case_id
            ]
            latest_number = max((attempt.get("number", 0) for _, attempt in attempt_values),
                                default=0)
            attempts = {
                identity for identity, attempt in attempt_values
                if attempt.get("number") == latest_number
            }
            sample_values = [
                sample for sample in projection["samples"].values()
                if sample["parent_id"] in attempts
            ]
            sample_values.sort(key=lambda sample: sample["number"])
            measurements = [measurement_from_payload(sample["measurement"])
                            for sample in sample_values]
            aggregate = aggregate_generation_measurements(measurements, case["requested_runs"])
            valid = [sample for sample in measurements if not measurement_validation_errors(sample)]
            if valid or sample_values:
                ttfts = [sample.client_ttft_sec for sample in valid]
                tps_values = [sample.tokens_per_sec for sample in valid]
                context_result = dict(aggregate)
                if valid:
                    import statistics
                    context_result.update({
                        "ttft_mean_sec": round(statistics.mean(ttfts), 3),
                        "ttft_stdev_sec": round(statistics.stdev(ttfts), 3) if len(ttfts) >= 2 else 0,
                        "tps_mean": round(statistics.mean(tps_values), 2),
                        "tps_stdev": round(statistics.stdev(tps_values), 2) if len(tps_values) >= 2 else 0,
                        "ttft_runs": [round(value, 3) for value in ttfts],
                        "tps_runs": [round(value, 2) for value in tps_values],
                    })
                if case.get("depth_tokens") is not None:
                    context_result["depth_tokens"] = case["depth_tokens"]
                context_result.update(case.get("result_fields", {}))
                results.setdefault(short, {})[case["context_label"]] = context_result
            results.setdefault(short, {}).update(case.get("model_markers", {}))
        return results


def export_llm_section(path: Path, job_id: str, stage_name: str = "llm",
                       model_family: str = "llm") -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = LLMEventStage(
        path, plan, lambda _: None, stage_name=stage_name,
        model_family=model_family, initialize=False,
    )
    try:
        return stage.export()
    finally:
        stage.close()
