"""Journal ownership and compatible projection for native llama-bench."""

from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan
from scripts.results.llm_event_stage import CaseTelemetryLike


def group_remaining_sweeps(pp, tg, completed):
    """Group pending native cases while preserving the two-sweep fresh-run shape."""
    sweeps = []
    pending_prefill = [depth for depth in pp if (depth, 0, 0) not in completed]
    if pending_prefill:
        sweeps.append(("prefill", pending_prefill, []))
    grouped = {}
    for depth in pp:
        pending_tg = tuple(tokens for tokens in tg if (0, tokens, depth) not in completed)
        if pending_tg:
            grouped.setdefault(pending_tg, []).append(depth)
    sweeps += [("decode", depths, list(tokens)) for tokens, depths in grouped.items()]
    return sweeps


class NativeBenchEventStage:
    def __init__(self, path: Path, plan: RunPlan, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 telemetry: CaseTelemetryLike | None = None):
        self.plan = plan
        self.store = EventStore(path)
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("llamabench")
        self.model_identities = {model.get("tag"): model for model in plan.models["llm"]}
        self.telemetry = telemetry
        if resume:
            if not initialize:
                raise ValueError("resume requires an initializing stage owner")
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.store.prepare_recovery(plan.job_id, "llamabench")
        elif initialize:
            self.store.start_stage(plan, "llamabench", resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self):
        self.store.close()

    def begin_model_load(self) -> None:
        if self.telemetry:
            self.telemetry.begin_model_load()

    def begin_measured(self, subwindow: str = "measured") -> None:
        if self.telemetry:
            self.telemetry.begin_measured(subwindow)

    def discard_case(self) -> None:
        if self.telemetry:
            self.telemetry.finish_case()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("llm", identity)

    def record_model_plan(self, model: dict, requested_cases: int, reps: int) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id("llamabench", model_id, {"model_plan": True})
        if case_id in self.store.rebuild(self.plan.job_id)["cases"]:
            return
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_plan", "model_short": model["short"],
                "requested_cases": requested_cases, "requested_reps": reps,
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, "complete", {}, parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def record_entry(self, model: dict, entry: dict) -> None:
        model_id = self._model_id(model)
        case_key = {key: entry.get(key, 0) for key in ("n_prompt", "n_gen", "n_depth")}
        case_id = self.plan.case_id("llamabench", model_id, case_key)
        attempt_id = self.plan.attempt_id(case_id, 1)
        sample_id = self.plan.sample_id(attempt_id, 1)
        memory = self.telemetry.finish_case() if self.telemetry else None
        if self.telemetry:
            self.telemetry.begin_measured("measured:native-sweep")
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "entry", "model_short": model["short"],
            }, parent_id=self.stage_id),
            JournalEvent("attempt", attempt_id, "running", {"number": 1}, parent_id=case_id),
            JournalEvent("sample", sample_id, "recorded", {
                "number": 1, "valid": True, "measurement": entry, "errors": [],
            }, parent_id=attempt_id),
            JournalEvent("attempt", attempt_id, "complete", {}, parent_id=case_id),
            JournalEvent("case", case_id, "complete", {"memory": memory}, parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        model_id = self._model_id(model)
        case_id = self.plan.case_id("llamabench", model_id, {"model_state": state})
        projection = self.store.rebuild(self.plan.job_id)
        events = []
        if case_id not in projection["cases"]:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
            }, parent_id=self.stage_id))
        elif projection["cases"][case_id]["state"] in {"complete", "skipped"}:
            return
        elif projection["cases"][case_id]["state"] != "running":
            raise ValueError("model-state case was not prepared for recovery")
        events.append(
            JournalEvent("case", case_id, state, {"model_result": result},
                         parent_id=self.stage_id)
        )
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def pending_sweeps(self, model: dict, pp: list[int], tg: list[int]):
        model_id = self._model_id(model)
        projection = self.store.rebuild(self.plan.job_id)
        completed = set()
        for depth in pp:
            keys = [(depth, 0, 0), *((0, tokens, depth) for tokens in tg)]
            for n_prompt, n_gen, n_depth in keys:
                case_id = self.plan.case_id(
                    "llamabench", model_id,
                    {"n_prompt": n_prompt, "n_gen": n_gen, "n_depth": n_depth},
                )
                if projection["cases"].get(case_id, {}).get("state") == "complete":
                    completed.add((n_prompt, n_gen, n_depth))
        return group_remaining_sweeps(pp, tg, completed)

    def finish(self) -> None:
        self.store.append(self.plan.job_id, [
            JournalEvent("stage", self.stage_id, "complete", {}, parent_id=self.plan.job_id),
        ])
        self.export_fn(self.export())

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        results = {}
        for case_id, case in projection["cases"].items():
            if case["parent_id"] != self.stage_id or case["state"] == "running":
                continue
            short = case["model_short"]
            if case["case_kind"] == "model_plan":
                results[short] = {
                    "prefill_entries": [], "decode_entries": [],
                    "requested_cases": case["requested_cases"], "completed_cases": 0,
                    "requested_repetitions": case["requested_cases"] * case["requested_reps"],
                    "completed_repetitions": 0,
                }
            elif case["case_kind"] == "model_state":
                results.setdefault(short, {}).update(case.get("model_result", {}))
            else:
                attempts = {identity for identity, attempt in projection["attempts"].items()
                            if attempt["parent_id"] == case_id}
                sample = next(value for value in projection["samples"].values()
                              if value["parent_id"] in attempts)
                entry = sample["measurement"]
                if case.get("memory") is not None:
                    entry = {**entry, "memory": {**case["memory"], "case_id": case_id}}
                model_result = results.setdefault(short, {
                    "prefill_entries": [], "decode_entries": [],
                    "requested_cases": 0, "completed_cases": 0,
                    "requested_repetitions": 0, "completed_repetitions": 0,
                })
                target = (model_result["prefill_entries"] if entry.get("n_gen", 0) == 0
                          else model_result["decode_entries"])
                target.append(entry)
                model_result["completed_repetitions"] += entry["completed_reps"]
                if entry["completed_reps"] == entry["requested_reps"]:
                    model_result["completed_cases"] += 1
        return results


def export_native_bench_section(path: Path, job_id: str) -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = NativeBenchEventStage(path, plan, lambda _: None, initialize=False)
    try:
        return stage.export()
    finally:
        stage.close()
