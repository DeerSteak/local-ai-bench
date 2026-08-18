"""Journal-owned vLLM offline benchmark cases and compatible projection."""

from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan


class VllmBenchEventStage:
    def __init__(self, path: Path, plan: RunPlan, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 selected_case_ids: list[str] | None = None):
        self.plan = plan
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("vllmbench")
        self.store = EventStore(path)
        self.model_identities = {model.get("tag"): model for model in plan.models["llm"]}
        self.recovery_attempts = {}
        if resume:
            if not initialize:
                raise ValueError("resume requires an initializing stage owner")
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.recovery_attempts = self.store.prepare_recovery(
                plan.job_id, "vllmbench", selected_case_ids,
            )
        elif initialize:
            self.store.start_stage(plan, "vllmbench", resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self) -> None:
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("llm", identity)

    def _case_id(self, model: dict, kind: str, input_len: int, output_len: int) -> str:
        if kind not in {"latency", "throughput"}:
            raise ValueError(f"invalid vLLM bench case kind: {kind}")
        return self.plan.case_id(
            "vllmbench", self._model_id(model),
            {"kind": kind, "input_len": input_len, "output_len": output_len},
        )

    def next_attempt(self, model: dict, kind: str, input_len: int,
                     output_len: int) -> int | None:
        case_id = self._case_id(model, kind, input_len, output_len)
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
        if projection["stages"].get(self.stage_id, {}).get("recovery_scope") == "selected":
            return None
        raise ValueError("incomplete vLLM bench case was not prepared for recovery")

    def record_case(self, model: dict, kind: str, input_len: int, output_len: int,
                    entry: dict | None, requested_cases: int, status: str = "ok", *,
                    attempt_number: int = 1, error: str | None = None) -> None:
        if status not in {"ok", "failed", "timed_out"}:
            raise ValueError(f"invalid vLLM bench status: {status}")
        if status == "ok" and not isinstance(entry, dict):
            raise ValueError("successful vLLM bench case requires an entry")
        case_id = self._case_id(model, kind, input_len, output_len)
        projection = self.store.rebuild(self.plan.job_id)
        existing = projection["cases"].get(case_id)
        if existing and existing["state"] in {"complete", "skipped"}:
            raise ValueError("vLLM bench case already completed")
        if existing and existing["state"] != "running":
            raise ValueError("incomplete vLLM bench case was not prepared for recovery")
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "vllm_bench_case", "model_short": model["short"],
                "model_label": model["label"], "kind": kind, "input_len": input_len,
                "output_len": output_len, "requested_cases": requested_cases,
            }, parent_id=self.stage_id))
        events.append(JournalEvent(
            "attempt", attempt_id, "running", {"number": attempt_number}, parent_id=case_id,
        ))
        if entry is not None:
            events.append(JournalEvent(
                "sample", self.plan.sample_id(attempt_id, 1), "recorded",
                {"number": 1, "valid": True, "measurement": {"entry": entry}, "errors": []},
                parent_id=attempt_id,
            ))
        terminal = "complete" if status == "ok" else status
        events.extend([
            JournalEvent("attempt", attempt_id, terminal, {}, parent_id=case_id),
            JournalEvent("case", case_id, terminal, {"error": error}, parent_id=self.stage_id),
        ])
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        if state not in {"skipped", "failed"}:
            raise ValueError(f"invalid vLLM bench model state: {state}")
        case_id = self.plan.case_id(
            "vllmbench", self._model_id(model), {"model_state": state},
        )
        if case_id in self.store.rebuild(self.plan.job_id)["cases"]:
            raise ValueError("vLLM bench model state already recorded")
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
                "model_label": model["label"],
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, state, {"model_result": result},
                         parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        records = {}
        model_states = {}
        for case_id, case in projection["cases"].items():
            if case["parent_id"] != self.stage_id or case["state"] == "running":
                continue
            short = case["model_short"]
            if case["case_kind"] == "model_state":
                model_states.setdefault(short, []).append(case)
                continue
            record = records.setdefault(short, {
                "latency_entries": [], "throughput_entries": [],
                "requested_cases": case["requested_cases"], "completed_cases": 0,
            })
            attempts = [
                (attempt_id, attempt) for attempt_id, attempt in projection["attempts"].items()
                if attempt["parent_id"] == case_id
            ]
            latest = max((attempt.get("number", 0) for _, attempt in attempts), default=0)
            attempt_ids = {attempt_id for attempt_id, attempt in attempts
                           if attempt.get("number") == latest}
            samples = [sample for sample in projection["samples"].values()
                       if sample["parent_id"] in attempt_ids]
            if samples:
                record[f"{case['kind']}_entries"].append(samples[-1]["measurement"]["entry"])
                record["completed_cases"] += 1
            if case["state"] == "timed_out":
                record.update(timed_out=True,
                              timed_out_at=f"{case['kind']} in{case['input_len']}")
            if case.get("error"):
                record["error"] = case["error"]
        for record in records.values():
            for key in ("latency_entries", "throughput_entries"):
                record[key].sort(key=lambda entry: (entry["input_len"], entry["output_len"]))
        for short, states in model_states.items():
            records.setdefault(short, {}).update(states[-1]["model_result"])
        return records

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


def export_vllm_bench(path: Path, job_id: str) -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = VllmBenchEventStage(path, plan, lambda _: None, initialize=False)
    try:
        return stage.export()
    finally:
        stage.close()
