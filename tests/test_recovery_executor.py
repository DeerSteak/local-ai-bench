import json

import pytest

from scripts.runtime.engines.base import GenerationMeasurement
from scripts.results.event_store import EventStore, JournalEvent
from scripts.results.llm_event_stage import LLMEventStage
from scripts.results.image_event_stage import ImageEventStage
from scripts.results.local_execution_context import (
    LocalExecutionContext, images_dir_for_result, load_local_execution_context,
    write_local_execution_context,
)
from scripts.results.recovery_executor import fork_journal_run, resume_journal_run, retry_selected_cases
from scripts.results.result_store import build_run_manifest, finish_run, finish_stage, start_stage
from scripts.results.run_plan import RunPlan


MODEL = {"tag": "model:4b", "short": "model", "label": "Model"}


def stopped_result(tmp_path, tests=("llm",)):
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=list(tests),
        stage_order=list(tests), models={
            "llm": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )
    result = tmp_path / "result.json"
    data = {"run": build_run_manifest(plan=plan, repo_root=tmp_path), "llm": {},
            "llm_conversation": {}, "llamabench": {}, "concurrency_tool": {},
            "concurrency_chat": {}}
    start_stage(data["run"], tests[0], 1)
    finish_stage(data["run"], tests[0], {}, "interrupted", "signal")
    finish_run(data["run"], "interrupted", "signal")
    result.write_text(json.dumps(data), encoding="utf-8")
    return result, plan


def test_recovery_executor_resumes_attempt_and_completes_original_result(tmp_path):
    result, plan = stopped_result(tmp_path)
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    path = result.with_suffix(".events.sqlite3")
    first = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    first.record_case(
        MODEL, 512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)],
        "timed_out", 1,
    )
    first.close()

    def runner(saved_plan, journal, stage_name, save, saved_identity, resume):
        assert (saved_plan, stage_name, saved_identity, resume) == (plan, "llm", identity, True)
        owner = LLMEventStage(
            journal, plan, lambda _: None, resume_identity=identity, resume=True,
        )
        owner.close()
        child = LLMEventStage(journal, plan, lambda _: None, initialize=False)
        assert child.next_context_attempt(MODEL, 512) == 2
        child.record_case(
            MODEL, 512, "512", [GenerationMeasurement(0.1, 120, 60, 2.1, 2.0)],
            "ok", 1, attempt_number=2,
        )
        child.finish()
        section = child.export()
        child.close()
        save(section)
        return section

    recovered = resume_journal_run(
        result, identity_builder=lambda _plan: identity, stage_runner=runner,
    )
    assert recovered["run"]["status"] == "complete"
    assert recovered["run"]["recovery_history"][0]["status"] == "interrupted"
    assert recovered["llm"]["model"]["512"]["tps_mean"] == 60
    assert json.loads(result.read_text())["run"]["status"] == "complete"
    journal = EventStore(path)
    assert journal.rebuild(plan.job_id)["jobs"][plan.job_id]["state"] == "complete"
    journal.close()


def test_recovery_executor_requires_journal_before_mutation(tmp_path):
    result, _ = stopped_result(tmp_path, ("llamabenchconc",))
    before = result.read_bytes()
    with pytest.raises(ValueError, match="no durable event journal"):
        resume_journal_run(result, identity_builder=lambda _plan: {})
    assert result.read_bytes() == before


def test_image_fork_rebinds_private_paths_to_new_job_and_output(tmp_path, monkeypatch):
    result, source_plan = stopped_result(tmp_path, ("img",))
    source_journal = result.with_suffix(".events.sqlite3")
    identity = {"plan_id": source_plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    owner = ImageEventStage(
        source_journal, source_plan, lambda _: None, resume_identity=identity,
    )
    owner.close()
    comfyui = (tmp_path / "Custom ComfyUI").resolve()
    write_local_execution_context(
        source_journal,
        LocalExecutionContext(source_plan.job_id, comfyui, (tmp_path / "old images").resolve()),
    )
    output = tmp_path / "results_forked.json"
    monkeypatch.setattr("scripts.results.recovery_executor.config.RESULTS_DIR", tmp_path)

    def runner(plan, journal, stage, save, saved_identity, resume):
        context = load_local_execution_context(journal, plan.job_id)
        assert context.comfyui_dir == comfyui
        assert context.images_dir == images_dir_for_result(output, tmp_path)
        child = ImageEventStage(
            journal, plan, save, resume_identity=saved_identity,
        )
        child.finish()
        child.close()
        return {}

    forked = fork_journal_run(
        result, output, identity_builder=lambda _plan: identity, stage_runner=runner,
    )
    assert forked["run"]["status"] == "complete"


def test_recovery_executor_requires_fork_before_mutation_on_identity_drift(tmp_path):
    result, plan = stopped_result(tmp_path)
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    stage = LLMEventStage(
        result.with_suffix(".events.sqlite3"), plan, lambda _: None,
        resume_identity=identity,
    )
    stage.close()
    before = result.read_bytes()
    changed = {**identity, "environment": {"profile_sha256": "changed"}}
    with pytest.raises(ValueError, match="fork required"):
        resume_journal_run(result, identity_builder=lambda _plan: changed)
    assert result.read_bytes() == before


def test_recovery_executor_finalizes_when_all_stage_evidence_already_completed(tmp_path):
    result, plan = stopped_result(tmp_path)
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    path = result.with_suffix(".events.sqlite3")
    stage = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    stage.finish()
    stage.close()
    journal = EventStore(path)
    journal.finish_job(plan.job_id, "interrupted", "signal_after_stage")
    journal.close()
    data = json.loads(result.read_text())
    data["run"]["stages"]["llm"]["status"] = "complete"
    data["run"]["stages"]["llm"]["finished_at"] = "finished"
    result.write_text(json.dumps(data), encoding="utf-8")

    recovered = resume_journal_run(
        result, identity_builder=lambda _plan: identity,
        stage_runner=lambda *_args: pytest.fail("completed stage reran"),
    )
    assert recovered["run"]["status"] == "complete"
    journal = EventStore(path)
    assert journal.rebuild(plan.job_id)["jobs"][plan.job_id]["state"] == "complete"
    journal.close()


def test_recovery_rebuilds_sidecar_for_already_completed_accuracy_stage(monkeypatch, tmp_path):
    result, plan = stopped_result(tmp_path, ("mcq",))
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    journal_path = result.with_suffix(".events.sqlite3")
    journal = EventStore(journal_path)
    journal.start_stage(plan, "mcq", identity)
    journal.append(plan.job_id, [JournalEvent(
        "stage", plan.stage_id("mcq"), "complete", {}, parent_id=plan.job_id,
    )])
    journal.finish_job(plan.job_id, "interrupted", "sidecar_export_failed")
    journal.close()
    data = json.loads(result.read_text())
    data["run"]["stages"]["mcq"]["status"] = "complete"
    result.write_text(json.dumps(data), encoding="utf-8")
    rebuilt = []
    monkeypatch.setattr(
        "scripts.results.recovery_executor._rebuild_accuracy_sidecar",
        lambda *args: rebuilt.append(args),
    )

    recovered = resume_journal_run(
        result, identity_builder=lambda _plan: identity,
        stage_runner=lambda *_args: pytest.fail("completed accuracy stage reran"),
    )
    assert recovered["run"]["status"] == "complete"
    assert rebuilt == [(result.resolve(), journal_path, plan, "mcq")]


def test_recovery_executor_records_user_interrupt_truthfully(tmp_path):
    result, plan = stopped_result(tmp_path)
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    stage = LLMEventStage(
        result.with_suffix(".events.sqlite3"), plan, lambda _: None,
        resume_identity=identity,
    )
    stage.close()

    def interrupt(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        resume_journal_run(
            result, identity_builder=lambda _plan: identity, stage_runner=interrupt,
        )
    recovered = json.loads(result.read_text())
    assert recovered["run"]["status"] == "interrupted"
    assert recovered["run"]["stages"]["llm"]["status"] == "interrupted"
    assert recovered["run"]["stages"]["llm"]["reason"] == "KeyboardInterrupt"


def test_recovery_executor_forks_plan_without_mutating_source(tmp_path):
    source, source_plan = stopped_result(tmp_path)
    source_data = json.loads(source.read_text())
    source_data.update({
        "bank_versions": {"mcq": "abc123"}, "sample_ids": {"mcq": ["q1"]},
        "accuracy_settings": {"timeout_seconds": 60},
    })
    source.write_text(json.dumps(source_data), encoding="utf-8")
    source_before = source.read_bytes()
    output = tmp_path / "fork.json"
    calls = []

    def identity_builder(plan):
        assert plan.plan_id == source_plan.plan_id
        assert plan.job_id != source_plan.job_id
        return {"plan_id": plan.plan_id}

    def runner(plan, journal, stage, save, identity, resume):
        calls.append((plan.job_id, journal, stage, identity, resume))
        owner = LLMEventStage(journal, plan, lambda _: None, resume_identity=identity)
        owner.finish()
        owner.close()
        section = {"model": {"512": {"tps_mean": 42.0}}}
        save(section)
        return section

    forked = fork_journal_run(
        source, output, identity_builder=identity_builder, stage_runner=runner,
    )
    assert source.read_bytes() == source_before
    assert forked["run"]["status"] == "complete"
    assert forked["run"]["plan_id"] == source_plan.plan_id
    assert forked["run"]["job_id"] != source_plan.job_id
    assert forked["run"]["forked_from"] == {
        "run_id": json.loads(source_before)["run"]["run_id"],
        "job_id": source_plan.job_id, "plan_id": source_plan.plan_id,
    }
    assert calls == [(
        forked["run"]["job_id"], output.with_suffix(".events.sqlite3"), "llm",
        {"plan_id": source_plan.plan_id}, False,
    )]
    assert json.loads(output.read_text())["llm"]["model"]["512"]["tps_mean"] == 42.0
    assert forked["bank_versions"] == {"mcq": "abc123"}
    assert forked["sample_ids"] == {"mcq": ["q1"]}
    assert forked["accuracy_settings"] == {"timeout_seconds": 60}
    journal = EventStore(output.with_suffix(".events.sqlite3"))
    assert journal.rebuild(forked["run"]["job_id"])["jobs"][forked["run"]["job_id"]]["state"] == "complete"
    journal.close()


def test_recovery_executor_fork_refuses_existing_output_before_identity_work(tmp_path):
    source, _ = stopped_result(tmp_path)
    output = tmp_path / "fork.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        fork_journal_run(
            source, output,
            identity_builder=lambda _plan: pytest.fail("identity should not be built"),
        )
    assert output.read_text(encoding="utf-8") == "keep"


def test_recovery_executor_retries_only_selected_case_and_keeps_run_incomplete(tmp_path):
    result, plan = stopped_result(tmp_path)
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    path = result.with_suffix(".events.sqlite3")
    first = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    for context in (512, 2048):
        first.record_case(MODEL, context, str(context), [], "timed_out", 1)
    first.close()
    model_id = plan.model_id("llm", plan.models["llm"][0])
    selected = plan.case_id("llm", model_id, {"context_tokens": 512})
    untouched = plan.case_id("llm", model_id, {"context_tokens": 2048})

    def runner(saved_plan, journal, stage, save, saved_identity, resume, selected_ids):
        owner = LLMEventStage(
            journal, saved_plan, lambda _: None, resume_identity=saved_identity,
            resume=resume, selected_case_ids=selected_ids,
        )
        owner.close()
        child = LLMEventStage(journal, saved_plan, lambda _: None, initialize=False)
        assert child.next_context_attempt(MODEL, 2048) is None
        child.record_case(
            MODEL, 512, "512", [GenerationMeasurement(0.1, 100, 50, 2.1, 2.0)],
            "ok", 1, attempt_number=2,
        )
        child.finish()
        section = child.export()
        child.close()
        save(section)
        return section

    retried = retry_selected_cases(
        result, [selected], identity_builder=lambda _plan: identity, stage_runner=runner,
    )
    assert retried["run"]["status"] == "failed"
    assert retried["run"]["reason"] == "selected_retry_has_remaining_cases"
    journal = EventStore(path)
    projection = journal.rebuild(plan.job_id)
    journal.close()
    assert projection["cases"][selected]["state"] == "complete"
    assert projection["cases"][untouched]["state"] == "timed_out"
    assert projection["jobs"][plan.job_id]["state"] == "failed"
    assert retried["llm"]["model"]["512"]["tps_mean"] == 50
