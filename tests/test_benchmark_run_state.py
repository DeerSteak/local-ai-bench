import json

import pytest

from scripts.app.benchmark import (
    checkpoint_terminal_exception, cleanup_signal_numbers, finish_event_job,
    fork_provenance, interruption_exit_code,
)
from scripts.results.event_store import EventStore
from scripts.app.orchestration import StageExecutionError
from scripts.results.result_store import build_run_manifest
from scripts.results.run_plan import RunPlan


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["emb"],
        stage_order=["emb"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": "embed", "short": "embed"}], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )


def test_interrupted_exception_checkpoints_pending_data_without_relabeling():
    results = {"run": {"status": "interrupted", "reason": "signal", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}, "llm": {"model": {"2K": {"tps": 10}}}}
    saved = []
    checkpoint_terminal_exception(results, SystemExit(), lambda label: saved.append((label, results.copy())))
    assert results["run"]["status"] == "interrupted"
    assert saved[0][0] == "run interrupted"
    assert saved[0][1]["llm"]["model"]["2K"]["tps"] == 10
    assert results["run"]["stages"]["llm"]["status"] == "interrupted"


def test_unhandled_exception_marks_run_failed():
    results = {"run": {"status": "running", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}}
    labels = []
    checkpoint_terminal_exception(results, RuntimeError(), labels.append)
    assert results["run"]["status"] == "failed"
    assert results["run"]["reason"] == "RuntimeError"
    assert labels == ["run failed"]
    assert results["run"]["stages"]["llm"]["status"] == "failed"


def test_nonfinite_exception_uses_specific_failure_reason():
    results = {"run": {"status": "running", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}}
    checkpoint_terminal_exception(
        results, ValueError("non-finite numeric value at $.llm.m.tps"), lambda label: None,
    )
    assert results["run"]["reason"] == "invalid_numeric_value"
    assert results["run"]["stages"]["llm"]["reason"] == "invalid_numeric_value"


def test_stage_failure_records_phase_specific_reason():
    results = {"run": {"status": "running", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}}
    exc = StageExecutionError("llm", "cleanup", RuntimeError("boom"))
    checkpoint_terminal_exception(results, exc, lambda _: None)
    assert results["run"]["reason"] == "stage_cleanup_failed"


def test_fork_provenance_requires_exact_plan_and_new_output(tmp_path):
    plan = make_plan()
    source = tmp_path / "source.json"
    manifest = build_run_manifest(plan=plan, repo_root=tmp_path)
    source.write_text(json.dumps({"run": manifest}), encoding="utf-8")
    output = tmp_path / "fork.json"
    assert fork_provenance(source, plan, output) == {
        "run_id": manifest["run_id"], "job_id": plan.job_id, "plan_id": plan.plan_id,
    }
    with pytest.raises(ValueError, match="must differ"):
        fork_provenance(source, plan, source)
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        fork_provenance(source, plan, output)


def test_fork_provenance_rejects_configuration_drift(tmp_path):
    plan = make_plan()
    changed = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["emb"],
        stage_order=["emb"], models=plan.models,
        effective_config={"warmup_runs": 1, "cpu_only": False, "force_all": False},
    )
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"run": {"plan": plan.to_dict()}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches"):
        fork_provenance(source, changed, tmp_path / "fork.json")


def test_finish_event_job_terminalizes_existing_journal_only(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"
    assert finish_event_job(path, plan, "complete") is False
    store = EventStore(path)
    store.start_stage(plan, "emb")
    store.close()
    assert finish_event_job(path, plan, "failed", "boom") is True
    store = EventStore(path)
    assert store.rebuild(plan.job_id)["jobs"][plan.job_id]["state"] == "failed"
    store.close()


def test_interruption_exit_code_uses_standard_signal_status():
    assert interruption_exit_code(2) == 130
    assert interruption_exit_code(15) == 143


def test_windows_sigbreak_uses_the_durable_cleanup_handler():
    class WindowsSignals:
        SIGINT = 2
        SIGTERM = 15
        SIGBREAK = 21

    assert cleanup_signal_numbers(WindowsSignals) == (2, 15, 21)
    assert interruption_exit_code(WindowsSignals.SIGBREAK) == 149
