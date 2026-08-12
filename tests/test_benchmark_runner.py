from types import SimpleNamespace

from scripts.app.benchmark import relay_runner_log, run_supervised_llm, run_supervised_stage
from scripts.runtime.engines.base import GenerationMeasurement
from scripts.results.llm_event_stage import LLMEventStage
from scripts.results.native_bench_event_stage import NativeBenchEventStage
from scripts.results.run_plan import RunPlan


def make_plan():
    return RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["llm"], stage_order=["llm"],
        models={
            "llm": [{"tag": "fake:model", "short": "fake"}],
            "concurrency": [], "embeddings": [], "images": [],
        },
        effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                          "force_all": False},
    )


def test_supervised_progress_log_keeps_machine_readable_prefix(capsys):
    line = ('::local-ai-bench-progress::{"kind":"model","stage":"llm",'
            '"status":"running","model":"Qwen 4B"}\n')
    relay_runner_log(line)
    assert capsys.readouterr().out == line


def test_relayed_log_keeps_the_runners_own_timestamp(capsys):
    """The runner stamps its own lines; a second stamp reports when the parent relayed
    the line, not when it happened, and the two can differ by seconds under buffering."""
    relay_runner_log("[13:48:18]   ->  Granite 4.1 3B: model supports 131072 ctx\n")
    out = capsys.readouterr().out
    assert out == "[13:48:18]   ->  Granite 4.1 3B: model supports 131072 ctx\n"
    assert out.count("[13:") == 1


def test_relayed_traceback_lines_are_not_stamped(capsys):
    """A runner's multi-line output arrives one line at a time; stamping each line
    turned a single warning into a column of timestamps."""
    relay_runner_log("(EngineCore pid=1) FileNotFoundError: ninja\n")
    assert capsys.readouterr().out == "(EngineCore pid=1) FileNotFoundError: ninja\n"


def test_relayed_log_is_still_redacted(capsys):
    from scripts.runtime.log_redaction import redact_log_text
    secret = "hf_" + "a" * 34
    relay_runner_log(f"downloading with token {secret}\n")
    out = capsys.readouterr().out
    assert out.rstrip() == redact_log_text(f"downloading with token {secret}")
    if redact_log_text(secret) != secret:
        assert secret not in out


def test_supervised_llm_checkpoints_commits_and_requires_clean_terminal(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"
    saved = []
    cancelled = []

    class Supervisor:
        def __init__(self, spec):
            assert spec.job_id == plan.job_id

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            sample = GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [sample], "ok", 1,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            cancelled.append(True)

    result = run_supervised_llm(plan, path, saved.append, Supervisor)
    assert result == saved[-1]
    assert saved[0]["fake"]["512"]["tps_mean"] == 50
    assert cancelled == [True]


def test_runner_failure_preserves_committed_case_for_parent_recovery(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"

    class CrashedSupervisor:
        def __init__(self, _spec):
            pass

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
            )
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "failed"})
            return 1

        @staticmethod
        def cancel():
            pass

    saved = []
    try:
        run_supervised_llm(plan, path, saved.append, CrashedSupervisor)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed runner was accepted")
    assert saved[-1]["fake"]["512"]["tps_mean"] == 50


def test_supervised_resume_prepares_durable_attempt_for_child_runner(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}, "environment": {}}
    first = LLMEventStage(path, plan, lambda _: None, resume_identity=identity)
    model = {"tag": "fake:model", "short": "fake", "label": "Fake"}
    first.record_case(
        model, 512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)],
        "timed_out", 1,
    )
    first.close()

    class Supervisor:
        def __init__(self, _spec):
            pass

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            assert stage.next_context_attempt(model, 512) == 2
            stage.record_case(
                model, 512, "512", [GenerationMeasurement(0.1, 120, 60, 2.1, 2.0)],
                "ok", 1, attempt_number=2,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            pass

    result = run_supervised_stage(
        plan, path, "llm", lambda _: None, Supervisor,
        resume_identity=identity, resume=True,
    )
    assert result["fake"]["512"]["tps_mean"] == 60


def test_parent_export_failure_keeps_child_commit(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"

    class InterruptedSupervisor:
        def __init__(self, _spec):
            pass

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
            )
            stage.close()
            callback({"kind": "event"})
            raise KeyboardInterrupt

        @staticmethod
        def cancel():
            pass

    def read_only_export(_section):
        raise OSError("read-only output")

    try:
        run_supervised_llm(plan, path, read_only_export, InterruptedSupervisor)
    except (KeyboardInterrupt, OSError):
        pass
    else:
        raise AssertionError("coordinator failure was swallowed")

    reopened = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        assert reopened.export()["fake"]["512"]["tps_mean"] == 50
    finally:
        reopened.close()


def test_parent_interruption_keeps_child_commit(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"

    class InterruptedSupervisor:
        def __init__(self, _spec):
            pass

        def run(self, _callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                512, "512", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
            )
            stage.close()
            raise KeyboardInterrupt

        @staticmethod
        def cancel():
            pass

    try:
        run_supervised_llm(plan, path, lambda _: None, InterruptedSupervisor)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("interruption was swallowed")
    reopened = LLMEventStage(path, plan, lambda _: None, initialize=False)
    try:
        assert reopened.export()["fake"]["512"]["tps_mean"] == 50
    finally:
        reopened.close()


def test_generic_supervisor_projects_conversation_stage(tmp_path):
    base = make_plan()
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["conv"],
        stage_order=["conv"], models=base.models,
        effective_config=base.effective_config,
    )
    path = tmp_path / "events.sqlite3"

    class Supervisor:
        def __init__(self, spec):
            assert spec.stage == "conv"

        def run(self, callback):
            stage = LLMEventStage(
                path.resolve(), plan, lambda _: None, stage_name="conv", initialize=False,
            )
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                0, "0K", [GenerationMeasurement(0.1, 96, 48, 2.1, 2.0)], "ok", 1,
                depth_tokens=400,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            pass

    result = run_supervised_stage(plan, path, "conv", lambda _: None, Supervisor)
    assert result["fake"]["0K"]["depth_tokens"] == 400


def test_generic_supervisor_projects_native_stage(tmp_path):
    base = make_plan()
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["llamabench"],
        stage_order=["llamabench"], models=base.models,
        effective_config=base.effective_config,
    )
    path = tmp_path / "events.sqlite3"

    class Supervisor:
        def __init__(self, spec):
            assert spec.stage == "llamabench"

        def run(self, callback):
            stage = NativeBenchEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            model = {"tag": "fake:model", "short": "fake", "label": "Fake"}
            stage.record_model_plan(model, 1, 2)
            stage.record_entry(model, {
                "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 100.0,
                "samples_ts": [99.0, 101.0], "ts_runs": [99.0, 101.0],
                "requested_reps": 2, "completed_reps": 2,
            })
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            pass

    result = run_supervised_stage(plan, path, "llamabench", lambda _: None, Supervisor)
    assert result["fake"]["completed_cases"] == 1


def test_generic_supervisor_projects_concurrency_model_family(tmp_path):
    base = make_plan()
    models = base.models
    models["llm"] = []
    models["concurrency"] = [{"tag": "fake:model", "short": "fake"}]
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["conc_tool"],
        stage_order=["conc_tool"], models=models, effective_config=base.effective_config,
    )
    path = tmp_path / "events.sqlite3"

    class Supervisor:
        def __init__(self, spec):
            assert spec.stage == "conc_tool"

        def run(self, callback):
            stage = LLMEventStage(
                path.resolve(), plan, lambda _: None, stage_name="conc_tool",
                model_family="concurrency", initialize=False,
            )
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                1, "1", [GenerationMeasurement(0.1, 50, 50, 1.1, 1.0)], "ok", 1,
                result_fields={"aggregate_tps": 45.0},
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            pass

    result = run_supervised_stage(plan, path, "conc_tool", lambda _: None, Supervisor)
    assert result["fake"]["1"]["aggregate_tps"] == 45


def test_runner_names_its_progress_events_with_the_plan_engine(monkeypatch, tmp_path):
    """A runner is a separate process, so it must set the progress engine itself —
    otherwise a multi-engine run's rows all land on the first engine."""
    from scripts.runtime import workload_runner
    from scripts.runtime import progress_events

    recorded = []
    monkeypatch.setattr(workload_runner, "set_progress_engine", recorded.append)
    monkeypatch.setattr(workload_runner, "load_runner_plan",
                        lambda path, job_id: SimpleNamespace(
                            engine_name="vllm", retry_crashed_models=False,
                            effective_config={"offline": False}))
    monkeypatch.setattr(workload_runner, "execute_llm_job", lambda *a, **k: None)
    monkeypatch.setenv("LOCAL_AI_BENCH_RUNNER_TOKEN", "token")
    store = tmp_path / "events.sqlite3"
    assert workload_runner.main(
        ["--job-id", "j1", "--stage", "llm", "--event-store", str(store)]
    ) == 0
    assert recorded == ["vllm"]
    assert progress_events is not None


def test_runner_reapplies_vllm_cache_policy_for_its_runtime_backend():
    from scripts.runtime.workload_runner import configure_runner_engine

    class Engine:
        configured = None

        @staticmethod
        def runtime_backend(hardware_backend, *, cpu_only=False):
            return "cpu" if cpu_only else hardware_backend

        def configure_kv_cache(self, runtime_backend):
            self.configured = runtime_backend
            return "fp8" if runtime_backend == "cuda" else "auto"

    engine = Engine()
    assert configure_runner_engine(engine, "cuda", False) == "fp8"
    assert engine.configured == "cuda"
    assert configure_runner_engine(engine, "cuda", True) == "auto"
    assert engine.configured == "cpu"


def test_runner_skips_cache_policy_for_engines_without_configuration_hook():
    from scripts.runtime.workload_runner import configure_runner_engine

    assert configure_runner_engine(object(), "cuda", False) == "auto"


def test_runner_restores_recorded_gpu_split_mode(monkeypatch):
    from scripts.runtime import config
    from scripts.runtime.workload_runner import apply_runner_settings

    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "layer")
    apply_runner_settings({"gpu_split_mode": "tensor"})
    assert config.LLAMACPP_GPU_SPLIT_MODE == "tensor"


def test_legacy_runner_plan_defaults_gpu_split_to_layer(monkeypatch):
    from scripts.runtime import config
    from scripts.runtime.workload_runner import apply_runner_settings

    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "tensor")
    apply_runner_settings({})
    assert config.LLAMACPP_GPU_SPLIT_MODE == "layer"
