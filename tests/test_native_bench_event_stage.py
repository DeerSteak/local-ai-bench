from scripts.results.native_bench_event_stage import (
    NativeBenchEventStage, export_native_bench_section, group_remaining_sweeps,
)
from scripts.results.run_plan import RunPlan


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


def test_remaining_native_sweeps_keep_fresh_run_compact_and_group_partial_recovery():
    assert group_remaining_sweeps([512, 2048], [128, 512], set()) == [
        ("prefill", [512, 2048], []),
        ("decode", [512, 2048], [128, 512]),
    ]
    completed = {(512, 0, 0), (0, 128, 512), (0, 128, 2048), (0, 512, 2048)}
    assert group_remaining_sweeps([512, 2048], [128, 512], completed) == [
        ("prefill", [2048], []), ("decode", [512], [512]),
    ]


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


def test_native_case_memory_survives_projection(tmp_path):
    memory = {
        "windows": [{"name": "measured", "sample_count": 1}],
        "summary": {"process_rss_gb": {"peak_gb": 3}},
        "headroom": {"absolute_gb": 5, "fraction": 0.5, "state": "comfortable"},
        "provenance": {"interval_sec": 1, "failed_samples": 0},
    }

    class Telemetry:
        def __init__(self):
            self.calls = []
            self.last_power = {
                "status": "recorded", "source": "nvidia-smi", "scope": "accelerator",
                "energy_joules": 10,
            }

        def begin_model_load(self):
            self.calls.append("load")

        def begin_measured(self, subwindow="measured"):
            self.calls.append(subwindow)

        def finish_case(self, ceiling_gb=None):
            self.calls.append("finish")
            return memory

    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    telemetry = Telemetry()
    stage = NativeBenchEventStage(path, plan, lambda _: None, telemetry=telemetry)
    stage.begin_measured("measured:native-sweep-includes-load")
    stage.record_entry(MODEL, entry())
    stage.discard_case()
    stage.close()
    projected = export_native_bench_section(path, plan.job_id)["model"]["prefill_entries"][0]
    assert {key: projected["memory"][key] for key in memory} == memory
    assert projected["memory"]["case_id"].startswith("case_")
    assert projected["power"]["energy_joules"] == 10
    assert projected["power"]["scope"] == "accelerator"
    assert projected["power"]["case_id"] == projected["memory"]["case_id"]
    assert projected["power"]["efficiency"]["unit"] == "tokens_per_joule"
    assert projected["power"]["efficiency"]["work_count"] > 0
    assert telemetry.calls == [
        "measured:native-sweep-includes-load", "finish", "measured:native-sweep",
        "finish",
    ]


def test_native_recovery_omits_completed_rows_and_reuses_model_plan(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = make_plan()
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    first = NativeBenchEventStage(path, plan, lambda _: None, resume_identity=identity)
    first.record_model_plan(MODEL, requested_cases=3, reps=2)
    first.record_entry(MODEL, entry())
    first.record_model_state(MODEL, "timed_out", {"timed_out": True})
    first.close()
    owner = NativeBenchEventStage(
        path, plan, lambda _: None, resume_identity=identity, resume=True,
    )
    owner.close()
    runner = NativeBenchEventStage(path, plan, lambda _: None, initialize=False)
    try:
        runner.record_model_plan(MODEL, requested_cases=3, reps=2)
        sweeps = runner.pending_sweeps(MODEL, [512, 2048], [128])
        assert sweeps == [("prefill", [2048], []), ("decode", [512, 2048], [128])]
        assert export_native_bench_section(path, plan.job_id)["model"]["requested_cases"] == 3
    finally:
        runner.close()
