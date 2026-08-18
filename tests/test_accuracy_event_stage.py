from scripts.results.accuracy_event_stage import AccuracyEventStage, export_accuracy
from scripts.results.run_plan import RunPlan
from scripts.workloads.mcq_benchmark import MCQBenchmark


MODEL = {"tag": "model:4b", "short": "model", "label": "Model 4B"}
QUESTIONS = [
    {"id": "q1", "category": "a", "choices": {"A": "x", "B": "y"}, "answer": "B"},
    {"id": "q2", "category": "b", "choices": {"A": "x", "B": "y"}, "answer": "A"},
    {"id": "q3", "category": "b", "choices": {"A": "x", "B": "y"}, "answer": "B"},
]


def make_plan():
    return RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=["mcq"],
        stage_order=["mcq"], models={
            "llm": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 3, "warmup_runs": 1, "cpu_only": False, "force_all": False,
        },
    )


def make_stage(path, plan=None, **kwargs):
    return AccuracyEventStage(
        path, plan or make_plan(), "mcq", QUESTIONS, "bank-v1", MCQBenchmark.score,
        lambda _results, _answers: None, **kwargs,
    )


def test_question_commit_rebuilds_scored_result_and_raw_sidecar(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = make_stage(path, plan)
    try:
        stage.record_question(MODEL, "q1", "B", "The answer is B", "ok")
        stage.record_question(
            MODEL, "q2", "B", "I think B", "timed_out",
            budget_nudged=True, likely_loop=True,
        )
        result = stage.export_results()["model"]
        answers = stage.export_answers()["model"]
    finally:
        stage.close()

    assert result["correct"] == 1 and result["answered"] == 2
    assert result["partial"] is True
    assert result["timed_out_ids"] == ["q2"]
    assert result["budget_nudged_ids"] == ["q2"]
    assert result["likely_loop_ids"] == ["q2"]
    assert [row["raw_response"] for row in answers["answers"]] == [
        "The answer is B", "I think B",
    ]
    projected_result, projected_answers = export_accuracy(
        path, plan.job_id, "mcq", QUESTIONS, "bank-v1", MCQBenchmark.score,
    )
    assert projected_result == {"model": result}
    assert projected_answers == {"model": answers}


def test_completed_question_is_not_pending_and_duplicate_commit_is_rejected(tmp_path):
    stage = make_stage(tmp_path / "events.sqlite3")
    try:
        stage.record_question(MODEL, "q1", "B", "B", "ok")
        assert [question["id"] for question in stage.pending_questions(MODEL)] == ["q2", "q3"]
        try:
            stage.record_question(MODEL, "q1", "B", "B again", "ok")
        except ValueError as exc:
            assert "already completed" in str(exc)
        else:
            raise AssertionError("duplicate accuracy question was accepted")
    finally:
        stage.close()


def test_failed_question_resumes_at_next_attempt_without_rerunning_completed_case(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = make_stage(path, plan)
    stage.record_question(MODEL, "q1", "B", "B", "ok")
    stage.record_question(MODEL, "q2", None, "", "crashed")
    stage.close()

    identity = {
        "plan_id": plan.plan_id, "artifacts": {}, "runtimes": {}, "methodology": {},
    }
    # The initial owner used EventStore's equivalent default identity.
    resumed = AccuracyEventStage(
        path, plan, "mcq", QUESTIONS, "bank-v1", MCQBenchmark.score,
        lambda _results, _answers: None, resume=True, resume_identity=identity,
    )
    try:
        assert resumed.next_attempt(MODEL, "q1") is None
        assert resumed.next_attempt(MODEL, "q2") == 2
        resumed.record_question(MODEL, "q2", "A", "A", "ok", attempt_number=2)
        assert resumed.export_results()["model"]["correct"] == 2
    finally:
        resumed.close()


def test_bank_hash_changes_case_identity_and_question_validation_is_strict(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = make_stage(path, plan)
    try:
        old_case = stage._case_id(MODEL, "q1")
    finally:
        stage.close()
    reopened = AccuracyEventStage(
        path, plan, "mcq", QUESTIONS, "bank-v2", MCQBenchmark.score,
        lambda _results, _answers: None, initialize=False,
    )
    try:
        assert reopened._case_id(MODEL, "q1") != old_case
    finally:
        reopened.close()

    duplicate = [QUESTIONS[0], {**QUESTIONS[1], "id": "q1"}]
    try:
        AccuracyEventStage(
            tmp_path / "other.sqlite3", make_plan(), "mcq", duplicate, "bank-v1",
            MCQBenchmark.score, lambda _results, _answers: None,
        )
    except ValueError as exc:
        assert "must be unique" in str(exc)
    else:
        raise AssertionError("duplicate question identities were accepted")


def test_model_skip_is_projected_without_fake_question_results(tmp_path):
    stage = make_stage(tmp_path / "events.sqlite3")
    try:
        stage.record_model_state(MODEL, "skipped", {
            "skipped": True, "skip_reason": "tool_calls_unsupported",
        })
        assert stage.export_results() == {
            "model": {
                "label": "Model 4B", "skipped": True,
                "skip_reason": "tool_calls_unsupported",
            },
        }
        assert stage.export_answers() == {}
    finally:
        stage.close()
