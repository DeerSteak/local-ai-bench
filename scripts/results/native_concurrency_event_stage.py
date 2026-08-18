"""Journal ownership and compatible projection for llama-batched-bench."""

from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan
from scripts.results.llm_event_stage import CaseTelemetryLike
from scripts.runtime.telemetry import add_power_efficiency


class NativeConcurrencyEventStage:
    def __init__(self, path: Path, plan: RunPlan, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 telemetry: CaseTelemetryLike | None = None):
        self.plan = plan
        self.store = EventStore(path)
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("llamabenchconc")
        self.model_identities = {model.get("tag"): model for model in plan.models["llm"]}
        self.telemetry = telemetry
        if resume:
            if not initialize:
                raise ValueError("resume requires an initializing stage owner")
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.store.prepare_recovery(plan.job_id, "llamabenchconc")
        elif initialize:
            self.store.start_stage(plan, "llamabenchconc", resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self) -> None:
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("llm", identity)

    def record_model_plan(self, model: dict, pp: int, ctx_size: int,
                          requested_cases: int) -> None:
        case_id = self.plan.case_id(
            "llamabenchconc", self._model_id(model), {"model_plan": True},
        )
        if case_id in self.store.rebuild(self.plan.job_id)["cases"]:
            return
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_plan", "model_short": model["short"],
                "pp": pp, "ctx_size": ctx_size, "requested_cases": requested_cases,
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, "complete", {}, parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def completed_keys(self, model: dict) -> set[tuple[int, int, int]]:
        model_id = self._model_id(model)
        projection = self.store.rebuild(self.plan.job_id)
        completed = set()
        for case in projection["cases"].values():
            if (case["parent_id"] == self.stage_id and case.get("model_id") == model_id
                    and case.get("case_kind") == "native_concurrency_entry"
                    and case["state"] == "complete"):
                completed.add((case["pp"], case["tg"], case["pl"]))
        return completed

    def begin_measured(self, subwindow: str) -> None:
        if self.telemetry:
            self.telemetry.begin_measured(subwindow)

    def discard_case(self) -> None:
        if self.telemetry:
            self.telemetry.finish_case()

    def record_entry(self, model: dict, entry: dict) -> bool:
        key = {name: entry.get(name, 0) for name in ("pp", "tg", "pl")}
        model_id = self._model_id(model)
        case_id = self.plan.case_id("llamabenchconc", model_id, key)
        if self.store.rebuild(self.plan.job_id)["cases"].get(case_id, {}).get("state") == "complete":
            return False
        attempt_id = self.plan.attempt_id(case_id, 1)
        memory = self.telemetry.finish_case() if self.telemetry else None
        power = getattr(self.telemetry, "last_power", None) if self.telemetry else None
        power = add_power_efficiency(
            power, "tokens_per_joule", entry.get("tg", 0) * entry.get("pl", 0),
        )
        if self.telemetry:
            self.telemetry.begin_measured("measured:native-sweep")
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "native_concurrency_entry", "model_short": model["short"],
                "model_id": model_id, **key,
            }, parent_id=self.stage_id),
            JournalEvent("attempt", attempt_id, "running", {"number": 1}, parent_id=case_id),
            JournalEvent("sample", self.plan.sample_id(attempt_id, 1), "recorded", {
                "number": 1, "valid": True, "measurement": entry, "errors": [],
            }, parent_id=attempt_id),
            JournalEvent("attempt", attempt_id, "complete", {}, parent_id=case_id),
            JournalEvent("case", case_id, "complete", {"memory": memory, "power": power},
                         parent_id=self.stage_id),
        ])
        self.export_fn(self.export())
        return True

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        if state not in {"failed", "skipped", "timed_out"}:
            raise ValueError(f"invalid native concurrency model state: {state}")
        case_id = self.plan.case_id(
            "llamabenchconc", self._model_id(model), {"model_state": state},
        )
        projection = self.store.rebuild(self.plan.job_id)
        if case_id in projection["cases"]:
            return
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, state, {"model_result": result},
                         parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

    def record_model_complete(self, model: dict) -> None:
        case_id = self.plan.case_id(
            "llamabenchconc", self._model_id(model), {"model_complete": True},
        )
        if case_id in self.store.rebuild(self.plan.job_id)["cases"]:
            return
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_complete", "model_short": model["short"],
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, "complete", {}, parent_id=self.stage_id),
        ])
        self.export_fn(self.export())

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
                    "entries": [], "pp": case["pp"], "ctx_size": case["ctx_size"],
                    "requested_cases": case["requested_cases"], "completed_cases": 0,
                }
            elif case["case_kind"] == "model_state":
                results.setdefault(short, {}).update(case.get("model_result", {}))
            elif case["case_kind"] == "model_complete":
                result = results.setdefault(short, {})
                result.pop("error", None)
                result.pop("timed_out", None)
            else:
                attempt_ids = {identity for identity, attempt in projection["attempts"].items()
                               if attempt["parent_id"] == case_id}
                sample = next(value for value in projection["samples"].values()
                              if value["parent_id"] in attempt_ids)
                entry = sample["measurement"]
                if case.get("memory") is not None:
                    entry = {**entry, "memory": {**case["memory"], "case_id": case_id}}
                if case.get("power") is not None:
                    entry = {**entry, "power": {**case["power"], "case_id": case_id}}
                result = results.setdefault(short, {"entries": [], "requested_cases": 0,
                                                     "completed_cases": 0})
                result["entries"].append(entry)
                result["completed_cases"] += 1
        for result in results.values():
            if "entries" in result:
                result["entries"].sort(key=lambda entry: (entry.get("tg", 0), entry.get("pl", 0)))
        return results


def export_native_concurrency(path: Path, job_id: str) -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = NativeConcurrencyEventStage(path, plan, lambda _: None, initialize=False)
    try:
        return stage.export()
    finally:
        stage.close()
