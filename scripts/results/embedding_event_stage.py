"""Journal-owned embedding input batches and compatible projection."""

import math
from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan
from scripts.runtime.engines.base import EmbeddingMeasurement, embedding_validation_errors
from scripts.runtime.shared import Shared


def embedding_corpus_hash(identity: dict) -> str:
    value = identity.get("artifacts", {}).get("corpus:embeddings", {}).get("sha256")
    if not isinstance(value, str) or not value:
        raise ValueError("embedding corpus identity is missing from the journal")
    return value


class EmbeddingEventStage:
    def __init__(self, path: Path, plan: RunPlan, corpus_hash: str, export_fn, *,
                 initialize: bool = True, resume_identity: dict | None = None,
                 resume: bool = False, selected_case_ids: list[str] | None = None):
        if not isinstance(corpus_hash, str) or not corpus_hash:
            raise ValueError("embedding stage requires an input-corpus hash")
        self.plan = plan
        self.corpus_hash = corpus_hash
        self.export_fn = export_fn
        self.stage_id = plan.stage_id("emb")
        self.store = EventStore(path)
        self.model_identities = {model.get("tag"): model for model in plan.models["embeddings"]}
        self.recovery_attempts = {}
        if resume:
            if not initialize:
                raise ValueError("resume requires an initializing stage owner")
            if self.store.load_plan(plan.job_id) != plan:
                raise ValueError("resume plan does not match the journal job")
            if resume_identity is None or self.store.resume_identity(plan.job_id) != resume_identity:
                raise ValueError("resume identity changed; create a fork")
            self.recovery_attempts = self.store.prepare_recovery(
                plan.job_id, "emb", selected_case_ids,
            )
        elif initialize:
            self.store.start_stage(plan, "emb", resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self) -> None:
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("embeddings", identity)

    def _case_id(self, model: dict) -> str:
        return self.plan.case_id(
            "emb", self._model_id(model), {"corpus_hash": self.corpus_hash, "batch": 1},
        )

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
        raise ValueError("incomplete embedding case was not prepared for recovery")

    def record_batch(self, model: dict, samples: list[EmbeddingMeasurement], status: str,
                     n_chunks: int, requested_runs: int, *, attempt_number: int = 1,
                     memory=None, power=None, crash_detail: str | None = None,
                     failure_result: dict | None = None) -> None:
        if status not in {"ok", "crashed", "timed_out", "failed"}:
            raise ValueError(f"unknown embedding batch status: {status}")
        if not isinstance(n_chunks, int) or isinstance(n_chunks, bool) or n_chunks < 1:
            raise ValueError("embedding batch requires a positive chunk count")
        case_id = self._case_id(model)
        existing = self.store.rebuild(self.plan.job_id)["cases"].get(case_id)
        if existing and existing["state"] in {"complete", "skipped"}:
            raise ValueError("embedding batch already completed")
        if existing and existing["state"] != "running":
            raise ValueError("incomplete embedding case was not prepared for recovery")
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "embedding_batch", "model_short": model["short"],
                "model_label": model["label"], "corpus_hash": self.corpus_hash,
                "n_chunks": n_chunks, "requested_runs": requested_runs,
            }, parent_id=self.stage_id))
        events.append(JournalEvent(
            "attempt", attempt_id, "running", {"number": attempt_number}, parent_id=case_id,
        ))
        for number, sample in enumerate(samples, 1):
            errors = embedding_validation_errors(sample)
            events.append(JournalEvent("sample", self.plan.sample_id(attempt_id, number),
                                       "recorded", {
                "number": number, "valid": not errors,
                "measurement": {
                    "client_wall_sec": sample.client_wall_sec
                    if isinstance(sample.client_wall_sec, (int, float))
                    and math.isfinite(sample.client_wall_sec) else None,
                    "model_load_sec": sample.model_load_sec
                    if isinstance(sample.model_load_sec, (int, float))
                    and math.isfinite(sample.model_load_sec) else None,
                },
                "errors": errors,
            }, parent_id=attempt_id))
        valid = any(not embedding_validation_errors(sample) for sample in samples)
        terminal = "complete" if valid else (
            "timed_out" if status == "timed_out" else "failed"
        )
        events.extend([
            JournalEvent("attempt", attempt_id, terminal, {}, parent_id=case_id),
            JournalEvent("case", case_id, terminal, {
                "run_status": status, "memory": memory, "power": power,
                "crash_detail": crash_detail, "failure_result": failure_result,
            }, parent_id=self.stage_id),
        ])
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        if state not in {"skipped", "failed"}:
            raise ValueError(f"invalid embedding model state: {state}")
        case_id = self.plan.case_id(
            "emb", self._model_id(model),
            {"corpus_hash": self.corpus_hash, "model_state": state},
        )
        existing = self.store.rebuild(self.plan.job_id)["cases"].get(case_id)
        if existing and existing["state"] != "running":
            return
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
                "model_label": model["label"], "corpus_hash": self.corpus_hash,
            }, parent_id=self.stage_id))
        events.append(JournalEvent("case", case_id, state, {"model_result": result},
                                   parent_id=self.stage_id))
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export())

    def export(self) -> dict:
        projection = self.store.rebuild(self.plan.job_id)
        results = {}
        for case_id, case in projection["cases"].items():
            if case["parent_id"] != self.stage_id or case["state"] == "running":
                continue
            short = case["model_short"]
            if case["case_kind"] == "model_state":
                results[short] = {"label": case["model_label"], **case["model_result"]}
                continue
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
            valid = [sample for sample in samples if sample["valid"]]
            if valid:
                walls = [sample["measurement"]["client_wall_sec"] for sample in valid]
                rates = [case["n_chunks"] / wall for wall in walls]
                result = {
                    "label": case["model_label"],
                    "chunks_per_sec_mean": round(Shared.mean(rates), 1),
                    "chunks_per_sec_stdev": round(Shared.stdev(rates), 1),
                    "device": "gpu", "n_chunks": case["n_chunks"],
                    "n_runs": len(samples), "requested_runs": case["requested_runs"],
                    "completed_runs": len(samples), "valid_runs": len(valid),
                    "invalid_runs": [
                        {"run": index + 1, "errors": sample["errors"]}
                        for index, sample in enumerate(samples) if not sample["valid"]
                    ],
                    "client_wall_runs_sec": [round(wall, 3) for wall in walls],
                    "runs": [round(rate, 1) for rate in rates],
                }
                if len(valid) >= 2:
                    result.update({
                        "chunks_per_sec_median": round(Shared.median(rates), 1),
                        "chunks_per_sec_cv": round(Shared.coefficient_of_variation(rates), 4),
                    })
            else:
                reason = case.get("run_status")
                result = case.get("failure_result") or {
                    "label": case["model_label"], "skipped": True,
                    "skip_reason": "known_crash" if reason == "crashed" else (
                        "timed_out" if reason == "timed_out" else "failed"
                    ),
                    "skip_detail": case.get("crash_detail") if reason == "crashed" else (
                        "Embedding run timed out (120s)" if reason == "timed_out"
                        else "All embedding runs failed"
                    ),
                }
            if case.get("memory") is not None:
                result["memory"] = case["memory"]
            if case.get("power") is not None:
                result["power"] = case["power"]
            results[short] = result
        return results

    def finish(self) -> None:
        projection = self.store.rebuild(self.plan.job_id)
        unresolved = any(
            case["parent_id"] == self.stage_id and case["state"] not in {"complete", "skipped"}
            for case in projection["cases"].values()
        )
        stage = projection["stages"].get(self.stage_id, {})
        state = "failed" if stage.get("recovery_scope") == "selected" and unresolved \
            else "complete"
        self.store.append(self.plan.job_id, [
            JournalEvent("stage", self.stage_id, state, {}, parent_id=self.plan.job_id),
        ])
        self.export_fn(self.export())


def export_embeddings(path: Path, job_id: str) -> dict:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
        corpus_hash = embedding_corpus_hash(store.resume_identity(job_id))
    finally:
        store.close()
    stage = EmbeddingEventStage(
        path, plan, corpus_hash, lambda _: None, initialize=False,
    )
    try:
        return stage.export()
    finally:
        stage.close()
