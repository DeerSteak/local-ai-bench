"""Journal-owned single-shot LLM cases and compatible JSON projection."""

from dataclasses import asdict
from pathlib import Path

from engines.base import (
    GenerationMeasurement, aggregate_generation_measurements, measurement_validation_errors,
)
from event_store import EventStore, JournalEvent
from run_plan import RunPlan


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
    def __init__(self, path: Path, plan: RunPlan, export_fn):
        self.plan = plan
        self.store = EventStore(path)
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("llm")
        self.model_identities = {model.get("tag"): model for model in plan.models["llm"]}
        self.store.create_job(plan)
        self.store.append(plan.job_id, [
            JournalEvent("job", plan.job_id, "running", {}),
            JournalEvent("stage", self.stage_id, "running", {"stage": "llm"},
                         parent_id=plan.job_id),
        ])

    def close(self):
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("llm", identity)

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id("llm", model_id, {"model_state": state})
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "model_short": model["short"], "model_label": model["label"],
                "case_kind": "model_state",
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, state, {"model_result": result},
                         parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def record_case(self, model: dict, context_tokens: int, context_label: str,
                    samples: list[GenerationMeasurement], status: str,
                    requested_runs: int, model_markers: dict | None = None) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id("llm", model_id, {"context_tokens": context_tokens})
        attempt_id = self.plan.attempt_id(case_id, 1)
        events = [
            JournalEvent("case", case_id, "running", {
                "model_short": model["short"], "model_label": model["label"],
                "case_kind": "context", "context_tokens": context_tokens,
                "context_label": context_label, "requested_runs": requested_runs,
            }, parent_id=self.stage_id),
            JournalEvent("attempt", attempt_id, "running", {"number": 1}, parent_id=case_id),
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
            }, parent_id=self.stage_id),
        ])
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def finish(self) -> None:
        models = self.export()
        with_results = sum(any(isinstance(value, dict) for value in model.values())
                           for model in models.values())
        self.store.append(self.plan.job_id, [
            JournalEvent("stage", self.stage_id, "complete", {
                "models_with_results": with_results,
            }, parent_id=self.plan.job_id),
        ])
        self.export_fn(models)

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        results = {}
        for case_id, case in projection["cases"].items():
            short = case["model_short"]
            if case["state"] == "running":
                continue
            if case["case_kind"] == "model_state":
                results.setdefault(short, {}).update(case.get("model_result", {}))
                continue
            attempts = {
                identity for identity, attempt in projection["attempts"].items()
                if attempt["parent_id"] == case_id
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
                results.setdefault(short, {})[case["context_label"]] = context_result
            results.setdefault(short, {}).update(case.get("model_markers", {}))
        return results
