import json

import pytest

from scripts.results.accuracy_event_stage import (
    AccuracyEventStage, export_accuracy, merge_memory_evidence, merge_power_evidence,
)
from scripts.results.recovery_inspector import inspect_recovery
from scripts.results.resume_policy import build_resume_identity
from scripts.results.run_plan import RunPlan
from scripts.workloads.mcq_benchmark import MCQBenchmark
from scripts.workloads.accuracy_registry import accuracy_spec


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

    assert result["answered"] == 2
    assert result["partial"] is True
    assert "accuracy_pct" not in result and "incorrect" not in result
    assert result["timed_out_ids"] == ["q2"]
    assert result["budget_nudged_ids"] == ["q2"]
    assert "likely_loop_ids" not in result
    assert [row["raw_response"] for row in answers["answers"]] == [
        "The answer is B", "I think B",
    ]
    projected_result, projected_answers = export_accuracy(
        path, plan.job_id, "mcq", QUESTIONS, "bank-v1", MCQBenchmark.score,
    )
    assert projected_result == {"model": result}
    assert projected_answers == {"model": answers}


def test_question_commits_reuse_incremental_projection_without_replaying_journal(
        tmp_path, monkeypatch):
    stage = make_stage(tmp_path / "events.sqlite3")
    replay_count = 0
    original_events = stage.store.events

    def counted_events(job_id):
        nonlocal replay_count
        replay_count += 1
        return original_events(job_id)

    monkeypatch.setattr(stage.store, "events", counted_events)
    for question in QUESTIONS:
        stage.record_question(MODEL, question["id"], question["answer"], "answer", "ok")
    assert replay_count == 0
    stage.close()


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


def test_model_skip_is_idempotent_when_resume_rechecks_missing_model(tmp_path):
    stage = make_stage(tmp_path / "events.sqlite3")
    result = {"skipped": True, "skip_reason": "model_not_downloaded"}
    stage.record_model_state(MODEL, "skipped", result)
    stage.record_model_state(MODEL, "skipped", result)
    assert stage.export_results()["model"] == {"label": "Model 4B", **result}
    stage.close()


def test_model_state_export_retains_memory_and_power_evidence(tmp_path):
    stage = make_stage(tmp_path / "events.sqlite3")
    memory = {"summary": {"process_rss_gb": {"peak_gb": 2.0}}}
    power = {"status": "recorded", "source": "sensor", "scope": "package"}
    stage.record_model_state(MODEL, "failed", {"crashed": True})
    stage.record_model_evidence(MODEL, memory, power)
    assert stage.export_results()["model"] == {
        "label": "Model 4B", "crashed": True, "memory": memory, "power": power,
    }
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
        resumed.record_question(MODEL, "q3", None, "", "ok")
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


def test_question_bank_content_change_forces_fork(tmp_path):
    result_path = tmp_path / "results.json"
    event_path = tmp_path / "results.events.sqlite3"
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(QUESTIONS))
    plan = make_plan()
    saved_identity = build_resume_identity(
        plan, artifacts={"bank:mcq": bank_path}, runtimes={}, methodology={},
    )
    stage = AccuracyEventStage(
        event_path, plan, "mcq", QUESTIONS, "bank-v1", MCQBenchmark.score,
        lambda _results, _answers: None, resume_identity=saved_identity,
    )
    stage.record_question(MODEL, "q1", "B", "B", "ok")
    stage.close()
    result_path.write_text(json.dumps({
        "run": {"status": "interrupted", "plan": plan.to_dict()},
    }))

    bank_path.write_text(json.dumps([{**QUESTIONS[0], "answer": "A"}, *QUESTIONS[1:]]))
    current_identity = build_resume_identity(
        plan, artifacts={"bank:mcq": bank_path}, runtimes={}, methodology={},
    )
    report = inspect_recovery(result_path, lambda _plan: current_identity)
    assert report["can_resume"] is False
    assert report["reasons"] == ["model artifacts identity changed"]


@pytest.mark.parametrize("stage_name", ["mcq", "math", "reasoning", "code", "tool"])
def test_every_accuracy_bank_projects_its_existing_score_shape(tmp_path, stage_name):
    spec = accuracy_spec(stage_name)
    questions = spec.benchmark.load_questions()[:2]
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp", tests=[stage_name],
        stage_order=[stage_name], models={
            "llm": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 3, "warmup_runs": 1, "cpu_only": False, "force_all": False,
        },
    )
    stage = AccuracyEventStage(
        tmp_path / f"{stage_name}.sqlite3", plan, stage_name, questions,
        "bank-v1", spec.benchmark.score, lambda _results, _answers: None,
    )
    try:
        for question in questions:
            stage.record_question(MODEL, question["id"], None, "", "ok")
        projected = stage.export_results()["model"]
    finally:
        stage.close()
    expected = spec.benchmark.score(
        questions, {question["id"]: None for question in questions},
    )
    expected.pop("all")
    assert projected == {"label": MODEL["label"], **expected}


def test_recovery_telemetry_segments_merge_without_losing_earlier_evidence():
    memory = []
    power = []
    for number, (samples, mean, peak, energy) in enumerate(((2, 4.0, 5.0, 10.0),
                                                            (3, 8.0, 9.0, 20.0)), 1):
        memory.append({
            "windows": [{"name": f"measured:q{number}", "sample_count": samples}],
            "summary": {"process_rss_gb": {
                "peak_gb": peak, "mean_gb": mean, "final_gb": mean,
                "valid_samples": samples,
            }},
            "headroom": {
                "absolute_gb": 10.0 - peak, "fraction": (10.0 - peak) / 10.0,
                "state": "comfortable", "basis_channel": "process_rss_gb",
            },
            "provenance": {
                "interval_sec": 0.5, "failed_samples": number - 1,
                "channels": {"process_rss_gb": {
                    "source": "psutil", "failed_samples": number - 1,
                }},
            },
        })
        power.append({
            "status": "recorded", "reason": None, "source": "sensor",
            "scope": "package", "energy_joules": energy, "mean_watts": mean,
            "peak_watts": peak, "idle_baseline_watts": 2.0,
            "windows": [{"name": "idle", "sample_count": 1},
                        {"name": f"measured:q{number}", "sample_count": samples}],
            "provenance": {"interval_sec": 0.5, "failed_samples": number - 1},
        })
    merged_memory = merge_memory_evidence(memory)
    merged_power = merge_power_evidence(power)
    assert merged_memory is not None and merged_power is not None
    assert merged_memory["summary"]["process_rss_gb"] == {
        "peak_gb": 9.0, "mean_gb": 6.4, "final_gb": 8.0, "valid_samples": 5,
    }
    assert merged_memory["headroom"]["absolute_gb"] == 1.0
    assert merged_memory["provenance"]["failed_samples"] == 1
    assert merged_power["energy_joules"] == 30.0
    assert merged_power["peak_watts"] == 9.0
    assert len(merged_power["windows"]) == 4


def test_recovery_telemetry_rejects_source_or_interval_drift():
    with pytest.raises(ValueError, match="memory interval changed"):
        merge_memory_evidence([
            {"summary": {}, "provenance": {"interval_sec": 0.5}},
            {"summary": {}, "provenance": {"interval_sec": 1.0}},
        ])
    with pytest.raises(ValueError, match="power source or scope changed"):
        merge_power_evidence([
            {"source": "a", "scope": "gpu"}, {"source": "b", "scope": "gpu"},
        ])
