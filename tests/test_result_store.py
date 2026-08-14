import json

import pytest

from scripts.results import result_store
from scripts.results.run_plan import RunPlan


def _plan():
    return RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llamabench"],
        stage_order=["llamabench"],
        models={"llm": [], "concurrency": [], "embeddings": [], "images": []},
        effective_config={"warmup_runs": 1, "cpu_only": False, "force_all": False},
    )


def test_atomic_write_json_replaces_existing_file(tmp_path):
    path = tmp_path / "nested" / "result.json"
    result_store.atomic_write_json(path, {"old": True})
    result_store.atomic_write_json(path, {"new": "✓"})
    assert json.loads(path.read_text()) == {"new": "✓"}
    assert not list(path.parent.glob("*.tmp"))


def test_atomic_write_json_preserves_destination_when_replace_fails(monkeypatch, tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"old": true}')
    monkeypatch.setattr(result_store.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        result_store.atomic_write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_json_reports_nonfinite_path(tmp_path):
    with pytest.raises(ValueError, match=r"\$\.models\[0\]\.tps"):
        result_store.atomic_write_json(tmp_path / "x.json", {"models": [{"tps": float("nan")}]})


def test_run_state_precedence_and_counts(monkeypatch):
    monkeypatch.setattr(result_store, "utc_now", lambda: "2026-01-01T00:00:00+00:00")
    run = {"stages": {}}
    result_store.start_stage(run, "llm", 3)
    section = {
        "ok": {"2K": {"tps_mean": 10}},
        "skip": {"skipped": True},
        "bad": {"crashed": "2K"},
    }
    result_store.finish_stage(run, "llm", section)
    assert run["stages"]["llm"] == {
        "status": "complete", "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:00+00:00", "selected_models": 3,
        "models_with_results": 1, "models_skipped": 1, "models_failed": 1,
    }


def test_model_counts_treats_crashed_llm_with_context_data_as_usable():
    section = {
        "qwen": {"2K": {"tps_mean": 20.0}, "8K": {"tps_mean": 18.0}, "crashed": "32K"},
        "skip": {"skipped": True, "skip_reason": "slow"},
        "failed": {"crashed": "2K"},
    }
    assert result_store.model_counts(section) == {
        "models_with_results": 1, "models_skipped": 1, "models_failed": 1,
    }


def test_model_counts_treats_crashed_accuracy_score_as_usable():
    section = {"q": {
        "accuracy_pct": 71.0, "correct": 10, "total": 14, "crashed": True,
    }}
    assert result_store.model_counts(section) == {
        "models_with_results": 1, "models_skipped": 0, "models_failed": 0,
    }


def test_model_counts_does_not_treat_diagnostics_as_measurements():
    section = {"q": {
        "crashed": True, "timed_out_count": 2, "timed_out_ids": ["q1", "q2"],
        "memory_at_failure": {"system_ram_used_gb": 12.0},
    }}
    assert result_store.model_counts(section)["models_failed"] == 1


def test_model_identity_excludes_paths_and_unknown_fields():
    assert result_store.model_identity([{
        "tag": "a", "short": "b", "size_gb": 4, "path": "/secret", "token": "x",
    }]) == [{"tag": "a", "short": "b", "size_gb": 4}]


def test_run_manifest_identifies_streamed_internal_llamabench_methodology(
        monkeypatch, tmp_path):
    monkeypatch.setattr(result_store, "source_identity", lambda _: {})
    plan = _plan()
    run = result_store.build_run_manifest(
        plan=plan, repo_root=tmp_path,
    )
    assert run["schema_version"] == 5
    assert run["llamabench_repetition_mode"] == "streamed_internal_repetitions"
    assert run["plan_id"] == plan.plan_id
    assert run["job_id"] == plan.job_id
    assert run["plan"] == plan.to_dict()


def test_finish_stage_counts_models_missing_from_section_as_failed():
    run = {"stages": {}}
    result_store.start_stage(run, "emb", 2)
    result_store.finish_stage(run, "emb", {})
    assert run["stages"]["emb"]["models_failed"] == 2


def test_finish_active_stage_terminalizes_only_running_stage(monkeypatch):
    monkeypatch.setattr(result_store, "utc_now", lambda: "done")
    run = {"stages": {
        "llm": {"status": "complete", "finished_at": "earlier"},
        "emb": {"status": "running", "finished_at": None},
    }}
    result_store.finish_active_stage(run, "failed", "RuntimeError")
    assert run["stages"]["llm"]["status"] == "complete"
    assert run["stages"]["emb"] == {
        "status": "failed", "finished_at": "done", "reason": "RuntimeError",
    }


def test_result_store_owns_section_updates_and_stage_transitions(tmp_path):
    writes = []
    data = {"run": {"status": "running", "stages": {}}, "llm": {}}
    store = result_store.ResultStore(
        tmp_path / "result.json", data, writer=lambda path, value: writes.append(path),
    )
    store.start_stage("llm", 1)
    store.update_section("llm", {"m": {"2K": {"tps_mean": 10}}})
    store.complete_stage("llm")
    store.finish("complete")
    assert len(writes) == 4
    assert data["run"]["status"] == "complete"
    assert data["run"]["stages"]["llm"]["models_with_results"] == 1


def test_result_store_rejects_illegal_transitions(tmp_path):
    data = {"run": {"status": "running", "stages": {}}, "llm": {}}
    store = result_store.ResultStore(tmp_path / "result.json", data, writer=lambda *_: None)
    with pytest.raises(ValueError, match="not running"):
        store.complete_stage("llm")
    store.finish("failed")
    with pytest.raises(ValueError, match="after the run ended"):
        store.start_stage("llm", 1)


def test_result_store_rejects_duplicate_running_stage_and_second_finish(tmp_path):
    data = {"run": {"status": "running", "stages": {}}, "llm": {}}
    store = result_store.ResultStore(tmp_path / "result.json", data, writer=lambda *_: None)
    store.start_stage("llm", 1)
    with pytest.raises(ValueError, match="already running"):
        store.start_stage("llm", 1)
    store.finish("failed")
    with pytest.raises(ValueError, match="already terminal"):
        store.finish("failed")


def test_result_store_recovery_retains_prior_terminal_run_and_stage_state(tmp_path):
    data = {"run": {"status": "running", "stages": {}}, "llm": {}}
    store = result_store.ResultStore(tmp_path / "result.json", data, writer=lambda *_: None)
    store.start_stage("llm", 1)
    store.complete_stage("llm", status="interrupted", reason="signal")
    store.finish("interrupted", "signal")
    store.begin_recovery()
    store.resume_stage("llm", 1)
    assert data["run"]["status"] == "running"
    assert data["run"]["recovery_history"] == [{
        "status": "interrupted", "finished_at": data["run"]["recovery_history"][0]["finished_at"],
        "reason": "signal",
    }]
    assert data["run"]["stages"]["llm"]["status"] == "running"
    assert data["run"]["stages"]["llm"]["recovery_history"][0]["status"] == "interrupted"


def test_result_store_recovery_rejects_running_or_complete_stage_reopen(tmp_path):
    data = {"run": {"status": "running", "stages": {}}, "llm": {}}
    store = result_store.ResultStore(tmp_path / "result.json", data, writer=lambda *_: None)
    with pytest.raises(ValueError, match="terminal run"):
        store.begin_recovery()
    store.start_stage("llm", 1)
    with pytest.raises(ValueError, match="not terminal"):
        store.resume_stage("llm", 1)
