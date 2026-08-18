"""Journal-owned accuracy questions and compatible result/answer projections."""

from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan


def _weighted(values: list[tuple[float | None, int]]) -> float | None:
    selected = [(value, weight) for value, weight in values if value is not None and weight > 0]
    total = sum(weight for _, weight in selected)
    return sum(value * weight for value, weight in selected) / total if total else None


def merge_memory_evidence(raw_blocks: list[dict | None]) -> dict | None:
    blocks: list[dict] = [block for block in raw_blocks if isinstance(block, dict)]
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]
    summaries = {}
    channels = {
        channel for block in blocks for channel in block.get("summary", {})
    }
    for channel in channels:
        values = [block.get("summary", {}).get(channel, {}) for block in blocks]
        valid = sum(value.get("valid_samples", 0) for value in values)
        peaks = [value.get("peak_gb") for value in values if value.get("peak_gb") is not None]
        finals = [value.get("final_gb") for value in values if value.get("final_gb") is not None]
        summaries[channel] = {
            "peak_gb": max(peaks) if peaks else None,
            "mean_gb": _weighted([
                (value.get("mean_gb"), value.get("valid_samples", 0)) for value in values
            ]),
            "final_gb": finals[-1] if finals else None,
            "valid_samples": valid,
        }
    known_headroom = [
        block.get("headroom", {}) for block in blocks
        if block.get("headroom", {}).get("absolute_gb") is not None
    ]
    headroom = min(known_headroom, key=lambda value: value["absolute_gb"]) \
        if known_headroom else blocks[-1].get("headroom", {})
    provenances = [block.get("provenance", {}) for block in blocks]
    channel_provenance = {}
    for channel in {
            name for provenance in provenances for name in provenance.get("channels", {})}:
        values = [provenance.get("channels", {}).get(channel, {}) for provenance in provenances]
        sources = {value.get("source") for value in values if value.get("source") is not None}
        if len(sources) > 1:
            raise ValueError(f"accuracy memory source changed across recovery: {channel}")
        channel_provenance[channel] = {
            "source": next(iter(sources), "unsupported"),
            "failed_samples": sum(value.get("failed_samples", 0) for value in values),
        }
    intervals = {value.get("interval_sec") for value in provenances}
    if len(intervals) > 1:
        raise ValueError("accuracy memory interval changed across recovery")
    return {
        "windows": [window for block in blocks for window in block.get("windows", [])],
        "summary": summaries, "headroom": headroom,
        "provenance": {
            "interval_sec": next(iter(intervals), None),
            "failed_samples": sum(value.get("failed_samples", 0) for value in provenances),
            "channels": channel_provenance,
        },
    }


def merge_power_evidence(raw_blocks: list[dict | None]) -> dict | None:
    blocks: list[dict] = [block for block in raw_blocks if isinstance(block, dict)]
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]
    identities = {(block.get("source"), block.get("scope")) for block in blocks}
    if len(identities) > 1:
        raise ValueError("accuracy power source or scope changed across recovery")
    weights = [sum(window.get("sample_count", 0) for window in block.get("windows", []))
               for block in blocks]
    idle_weights = [sum(window.get("sample_count", 0) for window in block.get("windows", [])
                        if window.get("name") == "idle") for block in blocks]
    peaks = [float(value) for block in blocks
             if isinstance((value := block.get("peak_watts")), (int, float))
             and not isinstance(value, bool)]
    recorded = all(block.get("status") == "recorded" for block in blocks)
    provenances = [block.get("provenance", {}) for block in blocks]
    intervals = {value.get("interval_sec") for value in provenances}
    if len(intervals) > 1:
        raise ValueError("accuracy power interval changed across recovery")
    source, scope = next(iter(identities))
    return {
        "status": "recorded" if recorded else "unavailable",
        "reason": None if recorded else next(
            (block.get("reason") for block in blocks if block.get("reason")),
            "incomplete power evidence across recovery",
        ),
        "source": source, "scope": scope,
        "energy_joules": sum(block["energy_joules"] for block in blocks) if recorded else None,
        "mean_watts": _weighted([
            (block.get("mean_watts"), weight) for block, weight in zip(blocks, weights)
        ]),
        "peak_watts": max(peaks) if peaks else None,
        "idle_baseline_watts": _weighted([
            (block.get("idle_baseline_watts"), weight)
            for block, weight in zip(blocks, idle_weights)
        ]),
        "windows": [window for block in blocks for window in block.get("windows", [])],
        "provenance": {
            "interval_sec": next(iter(intervals), None),
            "failed_samples": sum(value.get("failed_samples", 0) for value in provenances),
        },
    }


class AccuracyEventStage:
    def __init__(self, path: Path, plan: RunPlan, stage_name: str, questions: list[dict],
                 bank_hash: str, score_fn, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False,
                 selected_case_ids: list[str] | None = None):
        if stage_name not in {"mcq", "math", "reasoning", "code", "tool"}:
            raise ValueError(f"unsupported accuracy stage: {stage_name}")
        question_ids = [question.get("id") for question in questions]
        if any(not isinstance(question_id, str) or not question_id for question_id in question_ids):
            raise ValueError("accuracy questions require non-empty string identities")
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("accuracy question identities must be unique")
        if not isinstance(bank_hash, str) or not bank_hash:
            raise ValueError("accuracy stage requires a question-bank hash")
        self.plan = plan
        self.stage_name = stage_name
        self.stage_id = plan.stage_id(stage_name)
        self.questions = questions
        self.question_by_id = {question["id"]: question for question in questions}
        self.bank_hash = bank_hash
        self.score_fn = score_fn
        self.export_fn = export_fn
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
                plan.job_id, stage_name, selected_case_ids,
            )
        elif initialize:
            self.store.start_stage(plan, stage_name, resume_identity)
        elif self.store.load_plan(plan.job_id) != plan:
            raise ValueError("runner plan does not match the journal job")

    def close(self) -> None:
        self.store.close()

    def _model_id(self, model: dict) -> str:
        identity = self.model_identities.get(model["tag"])
        if identity is None:
            raise ValueError(f"model is absent from run plan: {model['tag']}")
        return self.plan.model_id("llm", identity)

    def _case_id(self, model: dict, question_id: str) -> str:
        if question_id not in self.question_by_id:
            raise ValueError(f"question is absent from the selected bank: {question_id}")
        return self.plan.case_id(
            self.stage_name, self._model_id(model),
            {"bank_hash": self.bank_hash, "question_id": question_id},
        )

    def next_attempt(self, model: dict, question_id: str) -> int | None:
        case_id = self._case_id(model, question_id)
        projection = self.store.rebuild(self.plan.job_id)
        case = projection["cases"].get(case_id)
        if case is None:
            return 1
        if case["state"] == "complete":
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
        raise ValueError("incomplete accuracy case was not prepared for recovery")

    def pending_questions(self, model: dict) -> list[dict]:
        return [
            question for question in self.questions
            if self.next_attempt(model, question["id"]) is not None
        ]

    def record_question(self, model: dict, question_id: str, given, raw_response: str,
                        status: str, *, budget_nudged: bool = False,
                        likely_loop: bool = False, attempt_number: int = 1) -> None:
        if status not in {"ok", "budget_exceeded", "timed_out", "loop_detected", "crashed"}:
            raise ValueError(f"unknown accuracy question status: {status}")
        if not isinstance(raw_response, str):
            raise ValueError("accuracy raw response must be text")
        case_id = self._case_id(model, question_id)
        projection = self.store.rebuild(self.plan.job_id)
        existing = projection["cases"].get(case_id)
        if existing and existing["state"] == "complete":
            raise ValueError(f"accuracy question already completed: {question_id}")
        if existing and existing["state"] != "running":
            raise ValueError("incomplete accuracy case was not prepared for recovery")
        attempt_id = self.plan.attempt_id(case_id, attempt_number)
        sample_id = self.plan.sample_id(attempt_id, 1)
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "accuracy_question", "model_short": model["short"],
                "model_label": model["label"], "question_id": question_id,
                "bank_hash": self.bank_hash,
            }, parent_id=self.stage_id))
        valid = status != "crashed"
        errors = [] if valid else ["engine_crashed"]
        terminal = "complete" if valid else "failed"
        events.extend([
            JournalEvent("attempt", attempt_id, "running", {"number": attempt_number},
                         parent_id=case_id),
            JournalEvent("sample", sample_id, "recorded", {
                "number": 1, "valid": valid,
                "measurement": {"given": given, "raw_response": raw_response},
                "errors": errors,
            }, parent_id=attempt_id),
            JournalEvent("attempt", attempt_id, terminal, {}, parent_id=case_id),
            JournalEvent("case", case_id, terminal, {
                "run_status": status, "budget_nudged": budget_nudged,
                "likely_loop": likely_loop,
            }, parent_id=self.stage_id),
        ])
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export_results(), self.export_answers())

    def record_model_state(self, model: dict, state: str, result: dict) -> None:
        if state not in {"complete", "skipped", "failed"}:
            raise ValueError(f"invalid accuracy model state: {state}")
        case_id = self.plan.case_id(
            self.stage_name, self._model_id(model),
            {"bank_hash": self.bank_hash, "model_state": state},
        )
        existing = self.store.rebuild(self.plan.job_id)["cases"].get(case_id)
        if existing and existing["state"] != "running":
            return
        events = []
        if existing is None:
            events.append(JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
                "model_label": model["label"], "bank_hash": self.bank_hash,
            }, parent_id=self.stage_id))
        events.append(JournalEvent("case", case_id, state, {"model_result": result},
                                   parent_id=self.stage_id))
        self.store.append(self.plan.job_id, events)
        self.export_fn(self.export_results(), self.export_answers())

    def record_model_evidence(self, model: dict, memory, power=None) -> None:
        projection = self.store.rebuild(self.plan.job_id)
        segments = sum(
            case.get("parent_id") == self.stage_id
            and case.get("case_kind") == "model_evidence"
            and case.get("model_short") == model["short"]
            for case in projection["cases"].values()
        )
        case_id = self.plan.case_id(
            self.stage_name, self._model_id(model),
            {"bank_hash": self.bank_hash, "model_evidence": segments + 1},
        )
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_evidence", "model_short": model["short"],
                "model_label": model["label"], "bank_hash": self.bank_hash,
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, "complete", {
                "memory": memory, "power": power,
            }, parent_id=self.stage_id),
        ])
        self.export_fn(self.export_results(), self.export_answers())

    def _model_records(self) -> dict[str, dict]:
        projection = self.store.rebuild(self.plan.job_id)
        attempts_by_case = {}
        for attempt_id, attempt in projection["attempts"].items():
            attempts_by_case.setdefault(attempt["parent_id"], []).append((attempt_id, attempt))
        samples_by_attempt = {}
        for sample in projection["samples"].values():
            samples_by_attempt.setdefault(sample["parent_id"], []).append(sample)
        records = {}
        for case_id, case in projection["cases"].items():
            if case["parent_id"] != self.stage_id or case["state"] == "running":
                continue
            model = records.setdefault(case["model_short"], {
                "label": case["model_label"], "questions": {}, "model_states": [],
                "evidence": [],
            })
            if case["case_kind"] == "model_state":
                model["model_states"].append(case)
                continue
            if case["case_kind"] == "model_evidence":
                model["evidence"].append(case)
                continue
            attempts = attempts_by_case.get(case_id, [])
            latest = max((attempt.get("number", 0) for _, attempt in attempts), default=0)
            attempt_ids = {
                attempt_id for attempt_id, attempt in attempts
                if attempt.get("number") == latest
            }
            samples = [sample for attempt_id in attempt_ids
                       for sample in samples_by_attempt.get(attempt_id, [])]
            sample = samples[-1] if samples else {"measurement": {}}
            model["questions"][case["question_id"]] = {
                **case, **sample.get("measurement", {}), "case_id": case_id,
            }
        return records

    def export_results(self) -> dict:
        results = {}
        for short, record in self._model_records().items():
            if record["model_states"]:
                result = {
                    "label": record["label"],
                    **record["model_states"][-1].get("model_result", {}),
                }
                self._merge_record_evidence(result, record["evidence"])
                results[short] = result
                continue
            questions = record["questions"]
            answers = {question_id: value.get("given") for question_id, value in questions.items()}
            scored = self.score_fn(self.questions, answers)
            scored.pop("all", None)
            failed = [value for value in questions.values() if value["state"] == "failed"]
            incomplete = len(questions) < len(self.questions)
            result = (
                {"label": record["label"], "partial": True,
                 "answered": len(questions), "total": len(self.questions)}
                if incomplete and not failed else {"label": record["label"], **scored}
            )
            diagnostics = {
                "timed_out": [qid for qid, value in questions.items()
                              if value.get("run_status") == "timed_out"],
                "budget_nudged": [qid for qid, value in questions.items()
                                  if value.get("budget_nudged")],
                "budget_exceeded": [qid for qid, value in questions.items()
                                   if value.get("run_status") == "budget_exceeded"],
                "likely_loop": [qid for qid, value in questions.items()
                                if value.get("likely_loop")],
            }
            incorrect_ids = {row["id"] for row in result.get("incorrect", [])}
            diagnostics["likely_loop"] = [
                qid for qid in diagnostics["likely_loop"] if qid in incorrect_ids
            ]
            for key, values in diagnostics.items():
                if values:
                    result[f"{key}_count"] = len(values)
                    result[f"{key}_ids"] = values
            if failed:
                result["crashed"] = True
                result["crashed_at"] = failed[-1]["question_id"]
            self._merge_record_evidence(result, record["evidence"])
            results[short] = result
        return results

    @staticmethod
    def _merge_record_evidence(result: dict, evidence: list[dict]) -> None:
        if not evidence:
            return
        memory = merge_memory_evidence([item.get("memory") for item in evidence])
        power = merge_power_evidence([item.get("power") for item in evidence])
        if memory is not None:
            result["memory"] = memory
        if power is not None:
            result["power"] = power

    def export_answers(self) -> dict:
        answers = {}
        for short, record in self._model_records().items():
            if record["model_states"]:
                continue
            questions = record["questions"]
            given = {question_id: value.get("given") for question_id, value in questions.items()}
            scored = self.score_fn(self.questions, given)
            answers[short] = {
                "label": record["label"],
                "answers": [
                    {**row, "raw_response": questions.get(row["id"], {}).get("raw_response", "")}
                    for row in scored["all"] if row["id"] in questions
                ],
            }
            if len(questions) < len(self.questions):
                answers[short]["partial"] = True
        return answers

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
        self.export_fn(self.export_results(), self.export_answers())


def export_accuracy(path: Path, job_id: str, stage_name: str, questions: list[dict],
                    bank_hash: str, score_fn) -> tuple[dict, dict]:
    store = EventStore(path)
    try:
        plan = store.load_plan(job_id)
    finally:
        store.close()
    stage = AccuracyEventStage(
        path, plan, stage_name, questions, bank_hash, score_fn,
        lambda _results, _answers: None, initialize=False,
    )
    try:
        return stage.export_results(), stage.export_answers()
    finally:
        stage.close()
