from native_bench_event_stage import NativeBenchEventStage, export_native_bench_section
from run_plan import RunPlan


MODEL = {"tag": "model:4b", "short": "model", "label": "Model 4B"}


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llamabench"],
        stage_order=["llamabench"], models={
            "llm": [{"tag": MODEL["tag"], "short": MODEL["short"]}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={"warmup_runs": 0, "cpu_only": False, "force_all": False},
    )


def entry(**overrides):
    value = {
        "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 100.0,
        "samples_ts": [99.0, 101.0], "ts_runs": [99.0, 101.0],
        "requested_reps": 2, "completed_reps": 2,
    }
    value.update(overrides)
    return value


def test_native_journal_projects_streamed_rows_and_partial_timeout(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    stage = NativeBenchEventStage(path, plan, lambda _: None)
    try:
        stage.record_model_plan(MODEL, requested_cases=3, reps=2)
        stage.record_entry(MODEL, entry())
        stage.record_entry(MODEL, entry(
            n_prompt=0, n_gen=128, n_depth=512, avg_ts=50.0,
            samples_ts=[49.0], ts_runs=[49.0], completed_reps=1,
        ))
        stage.record_model_state(MODEL, "timed_out", {
            "timed_out": True, "timed_out_at": "decode", "error": "idle timeout",
        })
        stage.finish()
    finally:
        stage.close()
    result = export_native_bench_section(path, plan.job_id)["model"]
    assert len(result["prefill_entries"]) == 1
    assert len(result["decode_entries"]) == 1
    assert result["requested_cases"] == 3
    assert result["completed_cases"] == 1
    assert result["completed_repetitions"] == 3
    assert result["timed_out_at"] == "decode"


def test_native_commit_survives_export_callback_failure(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    calls = [0]

    def fail_after_plan(_section):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("read-only output")

    stage = NativeBenchEventStage(path, plan, fail_after_plan)
    try:
        stage.record_model_plan(MODEL, requested_cases=1, reps=2)
        try:
            stage.record_entry(MODEL, entry())
        except OSError:
            pass
        else:
            raise AssertionError("failed export was accepted")
    finally:
        stage.close()
    assert export_native_bench_section(path, plan.job_id)["model"]["completed_cases"] == 1
