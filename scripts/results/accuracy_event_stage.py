"""Journal-owned accuracy questions and compatible result/answer projections."""

from pathlib import Path

from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.run_plan import RunPlan


FINAL_QUESTION_STATES = {"complete", "failed"}


class AccuracyEventStage:
    def __init__(self, path: Path, plan: RunPlan, stage_name: str, questions: list[dict],
                 bank_hash: str, score_fn, export_fn, *, initialize: bool = True,
                 resume_identity: dict | None = None, resume: bool = False):
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
            self.recovery_attempts = self.store.prepare_recovery(plan.job_id, stage_name)
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
        case = self.store.rebuild(self.plan.job_id)["cases"].get(case_id)
        if case is None:
            return 1
        if case["state"] == "complete":
            return None
        if case_id in self.recovery_attempts:
            return self.recovery_attempts[case_id]
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
        if case_id in self.store.rebuild(self.plan.job_id)["cases"]:
            raise ValueError("accuracy model state already recorded")
        self.store.append(self.plan.job_id, [
            JournalEvent("case", case_id, "running", {
                "case_kind": "model_state", "model_short": model["short"],
                "model_label": model["label"], "bank_hash": self.bank_hash,
            }, parent_id=self.stage_id),
            JournalEvent("case", case_id, state, {"model_result": result},
                         parent_id=self.stage_id),
        ])
        self.export_fn(self.export_results(), self.export_answers())

    def _model_records(self) -> dict[str, dict]:
        projection = self.store.rebuild(self.plan.job_id)
        records = {}
        for case_id, case in projection["cases"].items():
            if case["parent_id"] != self.stage_id or case["state"] == "running":
                continue
            model = records.setdefault(case["model_short"], {
                "label": case["model_label"], "questions": {}, "model_states": [],
            })
            if case["case_kind"] == "model_state":
                model["model_states"].append(case)
                continue
            attempts = [
                (attempt_id, attempt) for attempt_id, attempt in projection["attempts"].items()
                if attempt["parent_id"] == case_id
            ]
            latest = max((attempt.get("number", 0) for _, attempt in attempts), default=0)
            attempt_ids = {
                attempt_id for attempt_id, attempt in attempts
                if attempt.get("number") == latest
            }
            samples = [
                sample for sample in projection["samples"].values()
                if sample["parent_id"] in attempt_ids
            ]
            sample = samples[-1] if samples else {"measurement": {}}
            model["questions"][case["question_id"]] = {
                **case, **sample.get("measurement", {}), "case_id": case_id,
            }
        return records

    def export_results(self) -> dict:
        results = {}
        for short, record in self._model_records().items():
            if record["model_states"]:
                results[short] = {
                    "label": record["label"],
                    **record["model_states"][-1].get("model_result", {}),
                }
                continue
            questions = record["questions"]
            answers = {question_id: value.get("given") for question_id, value in questions.items()}
            scored = self.score_fn(self.questions, answers)
            scored.pop("all", None)
            result = {"label": record["label"], **scored}
            if len(questions) < len(self.questions):
                result["partial"] = True
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
            failed = [value for value in questions.values() if value["state"] == "failed"]
            if failed:
                result["crashed"] = True
                result["crashed_at"] = failed[-1]["question_id"]
            results[short] = result
        return results

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
        state = "failed" if unresolved else "complete"
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
