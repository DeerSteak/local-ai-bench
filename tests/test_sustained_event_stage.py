from scripts.results.event_store import EventStore
from scripts.results.run_plan import RunPlan
from scripts.results.sustained_event_stage import SustainedEventStage, export_sustained_section
from scripts.runtime.engines.base import GenerationMeasurement


MODEL = {"tag": "model:4b", "short": "model", "label": "Model 4B"}


def plan():
    return RunPlan.create(
        application_version="6.0", engine_name="llamacpp", tests=["sustained"],
        stage_order=["sustained"], models={
            "llm": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "cpu_only": False, "force_all": False,
            "sustained_duration_sec": 600, "sustained_window_sec": 10,
            "sustained_context_tokens": 2048, "ambient_temp_c": None,
            "temperature_telemetry": True, "temperature_telemetry_interval_sec": 0.5,
            "temperature_sources": {"gpu_die_c": "nvidia-smi"},
            "memory_telemetry": True, "memory_telemetry_interval_sec": 0.5,
        },
    )


def measurement(tokens=100, tps=50):
    return GenerationMeasurement(
        client_ttft_sec=0.1, generated_tokens=tokens, tokens_per_sec=tps,
        client_wall_sec=2.1, decode_sec=2, finish_reason="length",
    )


def test_requests_are_durable_samples_before_the_soak_case_completes(tmp_path):
    path = tmp_path / "events.sqlite3"
    stage = SustainedEventStage(path, plan(), lambda _: None)
    stage.begin_case(MODEL, 1)
    stage.record_request(MODEL, 1, 1, measurement(), 0, 2.1)
    projection = stage.store.rebuild(stage.plan.job_id)
    sample = next(iter(projection["samples"].values()))
    assert sample["valid"] is True
    assert sample["measurement"]["start_sec"] == 0
    assert sample["measurement"]["end_sec"] == 2.1
    assert sample["measurement"]["generated_tokens"] == 100
    assert stage.export() == {}
    stage.close()


def test_completed_result_projects_after_reopen_without_response_content(tmp_path):
    path = tmp_path / "events.sqlite3"
    active_plan = plan()
    stage = SustainedEventStage(path, active_plan, lambda _: None)
    stage.begin_case(MODEL, 1)
    stage.record_request(MODEL, 1, 1, measurement(), 0, 2.1)
    stage.complete_case(MODEL, 1, {
        "series": [{"timestamp_sec": 0, "tokens_per_sec": 50}],
        "analysis": {"performance": "stable"},
    })
    stage.finish()
    stage.close()
    assert export_sustained_section(path, active_plan.job_id) == {"model": {
        "series": [{"timestamp_sec": 0, "tokens_per_sec": 50}],
        "analysis": {"performance": "stable"},
    }}
    store = EventStore(path)
    try:
        assert "response_text" not in str(store.rebuild(active_plan.job_id)["samples"])
    finally:
        store.close()


def test_skipped_model_is_terminal_without_an_attempt(tmp_path):
    path = tmp_path / "events.sqlite3"
    active_plan = plan()
    stage = SustainedEventStage(path, active_plan, lambda _: None)
    stage.record_model_state(MODEL, "skipped", {"skipped": "context_unsupported"})
    stage.finish()
    assert stage.export() == {"model": {"skipped": "context_unsupported"}}
    projection = stage.store.rebuild(active_plan.job_id)
    assert projection["attempts"] == {}
    stage.close()


def test_recovery_retries_the_whole_incomplete_model_case(tmp_path):
    path = tmp_path / "events.sqlite3"
    active_plan = plan()
    identity = {"plan_id": active_plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    stage = SustainedEventStage(path, active_plan, lambda _: None, resume_identity=identity)
    stage.begin_case(MODEL, 1)
    stage.record_request(MODEL, 1, 1, measurement(), 0, 2.1)
    stage.close()
    resumed = SustainedEventStage(
        path, active_plan, lambda _: None, resume=True, resume_identity=identity,
    )
    assert resumed.next_attempt(MODEL) == 2
    resumed.close()
