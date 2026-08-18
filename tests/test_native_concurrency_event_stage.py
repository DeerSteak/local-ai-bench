from scripts.results.native_concurrency_event_stage import NativeConcurrencyEventStage
from scripts.results.run_plan import RunPlan


MODEL = {"tag": "model", "short": "model", "label": "Model"}


def make_plan():
    return RunPlan.create(
        application_version="6.0-pre7", engine_name="llamacpp",
        tests=["llamabenchconc"], stage_order=["llamabenchconc"], models={
            "llm": [{"tag": "model", "short": "model"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )


def test_native_concurrency_projects_streamed_rows_and_ignores_replayed_rows(tmp_path):
    stage = NativeConcurrencyEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    stage.record_model_plan(MODEL, 512, 2048, 2)
    first = {"pp": 512, "tg": 128, "pl": 1, "speed_tg": 10.0}
    second = {"pp": 512, "tg": 128, "pl": 2, "speed_tg": 18.0}
    assert stage.record_entry(MODEL, first) is True
    assert stage.record_entry(MODEL, first) is False
    assert stage.record_entry(MODEL, second) is True
    assert stage.export()["model"] == {
        "entries": [first, second], "pp": 512, "ctx_size": 2048,
        "requested_cases": 2, "completed_cases": 2,
    }
    stage.close()


def test_native_concurrency_resume_reports_completed_rows_across_interruption(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    stage = NativeConcurrencyEventStage(path, plan, lambda _: None, resume_identity=identity)
    stage.record_model_plan(MODEL, 512, 2048, 2)
    stage.record_entry(MODEL, {"pp": 512, "tg": 128, "pl": 1})
    stage.close()
    resumed = NativeConcurrencyEventStage(
        path, plan, lambda _: None, resume=True, resume_identity=identity,
    )
    assert resumed.completed_keys(MODEL) == {(512, 128, 1)}
    resumed.close()


def test_native_concurrency_failure_merges_with_completed_evidence(tmp_path):
    stage = NativeConcurrencyEventStage(tmp_path / "events.sqlite3", make_plan(), lambda _: None)
    stage.record_model_plan(MODEL, 512, 2048, 2)
    stage.record_entry(MODEL, {"pp": 512, "tg": 128, "pl": 1})
    stage.record_model_state(MODEL, "timed_out", {"timed_out": True, "error": "idle"})
    result = stage.export()["model"]
    assert result["completed_cases"] == 1
    assert result["timed_out"] is True and result["error"] == "idle"
    stage.record_model_complete(MODEL)
    result = stage.export()["model"]
    assert "timed_out" not in result and "error" not in result
    stage.close()
