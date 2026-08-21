import json
import io
import sys
from types import SimpleNamespace

from scripts.app.benchmark import (
    relay_runner_log, run_supervised_llm, run_supervised_stage,
    temperature_telemetry_requested,
)
from scripts.runtime.engines.base import EmbeddingMeasurement, GenerationMeasurement
from scripts.results.llm_event_stage import LLMEventStage
from scripts.results.accuracy_event_stage import AccuracyEventStage
from scripts.results.embedding_event_stage import EmbeddingEventStage, export_embeddings
from scripts.results.image_event_stage import ImageEventStage, export_images
from scripts.results.local_execution_context import (
    LocalExecutionContext, write_local_execution_context,
)
from scripts.runtime import supervised_stage
from scripts.runtime.progress_events import PROGRESS_PREFIX
from scripts.runtime.shared import Shared
from scripts.workloads.mcq_benchmark import MCQBenchmark
from scripts.results.native_bench_event_stage import NativeBenchEventStage
from scripts.results.native_concurrency_event_stage import NativeConcurrencyEventStage
from scripts.results.vllm_bench_event_stage import VllmBenchEventStage
from scripts.results.run_plan import RunPlan
from scripts.runtime.telemetry import PowerAvailability, TemperatureAvailability
from scripts.runtime.workload_runner import (
    create_case_telemetry, execute_embedding_job, execute_image_job,
    inherited_power_availability,
    inherited_temperature_availability,
)


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


def test_qualification_temperature_override_controls_default_sustained_sampling():
    assert temperature_telemetry_requested(["llm"], {}) is False
    assert temperature_telemetry_requested(["sustained"], {}) is True
    assert temperature_telemetry_requested(
        ["sustained"], {"LOCAL_AI_BENCH_QUALIFICATION_TEMPERATURE": "0"},
    ) is False
    assert temperature_telemetry_requested(
        ["llm"], {"LOCAL_AI_BENCH_QUALIFICATION_TEMPERATURE": "1"},
    ) is True


def test_runner_power_telemetry_inherits_parent_source_inside_supervised_process(monkeypatch):
    status = PowerAvailability(True, "nvidia-smi", "accelerator", location="/tool")
    calls = []

    class Telemetry:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def start(self):
            calls.append(("start", {}))
            return self

    monkeypatch.setattr("scripts.runtime.workload_runner.CaseTelemetry", Telemetry)
    assert create_case_telemetry({"memory_telemetry": False, "power_telemetry": False}) is None
    telemetry = create_case_telemetry(
        {"memory_telemetry": True, "power_telemetry": True,
         "power_source": "nvidia-smi", "power_scope": "accelerator"},
        {"LOCAL_AI_BENCH_POWER_AVAILABILITY": json.dumps(status.__dict__)},
    )
    assert isinstance(telemetry, Telemetry)
    assert calls == [("init", {"power_availability": status}), ("start", {})]


def test_runner_refuses_inherited_power_identity_that_differs_from_plan():
    inherited = PowerAvailability(True, "rapl", "cpu_package", location="/counter")
    status = inherited_power_availability(
        {"power_source": "nvidia-smi", "power_scope": "accelerator"},
        {"LOCAL_AI_BENCH_POWER_AVAILABILITY": json.dumps(inherited.__dict__)},
    )
    assert status == PowerAvailability(
        False, "nvidia-smi", "accelerator",
        "parent power source was not inherited by the supervised process",
    )


def test_runner_inherits_only_the_planned_temperature_channels():
    status = TemperatureAvailability(
        True, {"gpu_die_c": "nvidia-smi"},
        locations={"gpu_die_c": "/usr/bin/nvidia-smi"},
    )
    inherited = inherited_temperature_availability(
        {"temperature_sources": {"gpu_die_c": "nvidia-smi"}},
        {"LOCAL_AI_BENCH_TEMPERATURE_AVAILABILITY": json.dumps(status.__dict__)},
    )
    assert inherited == status
    refused = inherited_temperature_availability(
        {"temperature_sources": {"cpu_package_c": "hwmon"}},
        {"LOCAL_AI_BENCH_TEMPERATURE_AVAILABILITY": json.dumps(status.__dict__)},
    )
    assert refused.available is False
    assert refused.sources == {"cpu_package_c": "hwmon"}
    assert refused.locations is None


def test_supervised_progress_log_keeps_machine_readable_prefix(capsys):
    line = ('::local-ai-bench-progress::{"kind":"model","stage":"llm",'
            '"status":"running","model":"Qwen 4B"}\n')
    relay_runner_log(line)
    assert capsys.readouterr().out == line


def test_supervised_progress_log_redacts_secrets_and_private_paths(capsys):
    secret = "hf_" + "a" * 34
    line = PROGRESS_PREFIX + json.dumps({
        "kind": "result",
        "stage": "run",
        "status": "complete",
        "path": f"C:\\Users\\Ben\\results_{secret}.json",
    }, separators=(",", ":"))

    relay_runner_log(line)

    output = capsys.readouterr().out.strip()
    assert output.startswith(PROGRESS_PREFIX)
    payload = json.loads(output.removeprefix(PROGRESS_PREFIX))
    assert payload["path"] == "<home>\\results_<secret>.json"
    assert secret not in output


def test_malformed_progress_log_is_still_redacted(capsys):
    secret = "hf_" + "a" * 34

    relay_runner_log(f"{PROGRESS_PREFIX}not-json token={secret}\n")

    output = capsys.readouterr().out
    assert output.startswith(PROGRESS_PREFIX)
    assert secret not in output


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


def test_relayed_log_never_crashes_a_legacy_windows_console(monkeypatch):
    output = io.BytesIO()
    console = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)
    relay_runner_log("[23:09:27] ✓ model loaded\n")
    console.flush()
    assert output.getvalue().decode("cp1252") == "[23:09:27] ? model loaded\n"


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


def test_llm_supervisor_receives_temperature_availability(tmp_path):
    plan = make_plan()
    path = tmp_path / "events.sqlite3"
    availability = TemperatureAvailability(True, {"gpu_die_c": "nvidia-smi"})

    class Supervisor:
        def __init__(self, spec):
            assert spec.temperature_availability == availability

        def run(self, callback):
            stage = LLMEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.finish()
            stage.close()
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel():
            pass

    run_supervised_llm(
        plan, path, lambda _section: None, Supervisor,
        temperature_availability=availability,
    )


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


def test_generic_supervisor_projects_vllm_bench_stage(tmp_path):
    base = make_plan()
    plan = RunPlan.create(
        application_version="4.1", engine_name="vllm", tests=["vllmbench"],
        stage_order=["vllmbench"], models=base.models,
        effective_config=base.effective_config,
    )
    path = tmp_path / "events.sqlite3"

    class Supervisor:
        def __init__(self, spec):
            assert spec.stage == "vllmbench"

        def run(self, callback):
            stage = VllmBenchEventStage(path.resolve(), plan, lambda _: None, initialize=False)
            stage.record_case(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                "latency", 512, 128,
                {"input_len": 512, "output_len": 128, "avg_latency_sec": 1.0}, 1,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel(): pass

    result = run_supervised_stage(plan, path, "vllmbench", lambda _: None, Supervisor)
    assert result["fake"]["completed_cases"] == 1


def test_generic_supervisor_projects_native_concurrency_stage(tmp_path):
    base = make_plan()
    plan = RunPlan.create(
        application_version="4.1", engine_name="llamacpp", tests=["llamabenchconc"],
        stage_order=["llamabenchconc"], models=base.models,
        effective_config=base.effective_config,
    )
    path = tmp_path / "events.sqlite3"

    class Supervisor:
        def __init__(self, spec): assert spec.stage == "llamabenchconc"

        def run(self, callback):
            stage = NativeConcurrencyEventStage(
                path.resolve(), plan, lambda _: None, initialize=False,
            )
            model = {"tag": "fake:model", "short": "fake", "label": "Fake"}
            stage.record_model_plan(model, 512, 2048, 1)
            stage.record_entry(model, {"pp": 512, "tg": 128, "pl": 1})
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel(): pass

    result = run_supervised_stage(
        plan, path, "llamabenchconc", lambda _: None, Supervisor,
    )
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


def test_supervised_embedding_projects_committed_batch(tmp_path):
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["emb"],
        stage_order=["emb"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": "embed:model", "short": "embed"}], "images": [],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "cpu_only": False, "force_all": False,
        },
    )
    identity = {
        "plan_id": plan.plan_id,
        "artifacts": {"corpus:embeddings": {"sha256": "corpus", "size": 1}},
        "runtimes": {}, "methodology": {},
    }
    path = (tmp_path / "results.events.sqlite3").resolve()

    class Supervisor:
        def __init__(self, spec): assert spec.stage == "emb"

        def run(self, callback):
            stage = EmbeddingEventStage(
                path, plan, "corpus", lambda _: None, initialize=False,
            )
            stage.record_batch(
                {"tag": "embed:model", "short": "embed", "label": "Embed"},
                [EmbeddingMeasurement([], 0.5)], "ok", 2, 1,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel(): pass

    result = run_supervised_stage(
        plan, path, "emb", lambda _: None, Supervisor, resume_identity=identity,
    )
    assert result["embed"]["chunks_per_sec_mean"] == 4.0


def test_supervised_image_projects_committed_resolution(tmp_path):
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "sdxl"}],
        }, effective_config={"runs": 1, "warmup_runs": 0, "cpu_only": False,
                             "force_all": False},
    )
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    path = (tmp_path / "results.events.sqlite3").resolve()

    class Supervisor:
        def __init__(self, spec): assert spec.stage == "img"

        def run(self, callback):
            stage = ImageEventStage(path, plan, lambda _: None, initialize=False)
            stage.record_resolution(
                {"short": "sdxl", "label": "SDXL", "checkpoint": "sdxl", "steps": 20},
                1024, 1024, [2.0], 1,
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel(): pass

    result = run_supervised_stage(
        plan, path, "img", lambda _: None, Supervisor, resume_identity=identity,
    )
    assert result["sdxl"]["resolutions"]["1024x1024"]["sec_per_image_mean"] == 2.0


def test_embedding_runner_reconstructs_plan_and_commits_projection(monkeypatch, tmp_path):
    from scripts.runtime import workload_runner

    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["emb"],
        stage_order=["emb"], models={
            "llm": [], "concurrency": [],
            "embeddings": [{"tag": "nomic-embed-text", "short": "nomic"}], "images": [],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "run_timeout_seconds": 7,
            "accuracy_timeout_seconds": 60, "accuracy_token_budget": 256,
            "cpu_only": False, "force_all": False, "max_prompt_tokens": None,
            "context_lengths": [512], "llamabench_pp": [512], "llamabench_tg": [128],
            "sample_size": None, "concurrency_tool_levels": [1],
            "concurrency_chat_levels": [1], "concurrency_tool_context": 512,
            "concurrency_chat_context": 1024, "concurrency_chat_soft_exit_floor": 1,
        },
    )
    identity = {
        "plan_id": plan.plan_id,
        "artifacts": {"corpus:embeddings": {"sha256": "corpus", "size": 1}},
        "runtimes": {}, "methodology": {},
    }
    path = tmp_path / "events.sqlite3"
    owner = EmbeddingEventStage(path, plan, "corpus", lambda _: None,
                                resume_identity=identity)
    owner.close()

    class Engine:
        def start(self, **_kwargs): return True
        def available(self): return False

    class Benchmark:
        def run(self, *, models, journal, **_kwargs):
            assert models[0]["label"] == "Nomic Embed Text"
            journal.record_batch(
                models[0], [EmbeddingMeasurement([], 0.5)], "ok", 2, 1,
            )
            journal.finish()

    monkeypatch.setattr(workload_runner, "apply_offline_mode", lambda _value: None)
    monkeypatch.setattr(workload_runner, "configure_runner_engine", lambda *_args: None)
    monkeypatch.setattr(workload_runner, "create_case_telemetry", lambda _settings: None)
    monkeypatch.setattr(workload_runner.Shared, "detect_backend", lambda: "cpu")
    monkeypatch.setattr(workload_runner.Shared, "shutdown_managed", lambda: None)
    monkeypatch.setattr(workload_runner, "emit", lambda *_args, **_kwargs: None)
    execute_embedding_job(
        path, plan.job_id, engine_factory=lambda _name: Engine(),
        benchmark_factory=lambda: Benchmark(),
    )
    assert export_embeddings(path, plan.job_id)["nomic"][
        "chunks_per_sec_mean"
    ] == 4.0


def test_image_runner_uses_private_paths_and_commits_projection(monkeypatch, tmp_path):
    from scripts.runtime import workload_runner

    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["img"],
        stage_order=["img"], models={
            "llm": [], "concurrency": [], "embeddings": [], "images": [{"short": "sdxl"}],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "run_timeout_seconds": 7,
            "accuracy_timeout_seconds": 60, "accuracy_token_budget": 256,
            "cpu_only": False, "force_all": False, "max_prompt_tokens": None,
            "context_lengths": [512], "llamabench_pp": [512], "llamabench_tg": [128],
            "sample_size": None, "concurrency_tool_levels": [1],
            "concurrency_chat_levels": [1], "concurrency_tool_context": 512,
            "concurrency_chat_context": 1024, "concurrency_chat_soft_exit_floor": 1,
        },
    )
    identity = {"plan_id": plan.plan_id, "artifacts": {}, "runtimes": {},
                "methodology": {}}
    path = tmp_path / "events.sqlite3"
    owner = ImageEventStage(path, plan, lambda _: None, resume_identity=identity)
    owner.close()
    comfyui = (tmp_path / "Custom ComfyUI").resolve()
    images = (tmp_path / "Custom Images").resolve()
    write_local_execution_context(
        path, LocalExecutionContext(plan.job_id, comfyui, images),
    )

    class Benchmark:
        def run(self, *, image_models, comfyui_dir, images_dir, journal, **_kwargs):
            assert image_models[0]["label"] == "SDXL"
            assert (comfyui_dir, images_dir) == (comfyui, images)
            journal.record_resolution(image_models[0], 1024, 1024, [2.0], 1)
            journal.finish()

    monkeypatch.setattr(workload_runner, "apply_offline_mode", lambda _value: None)
    monkeypatch.setattr(workload_runner, "create_case_telemetry", lambda _settings: None)
    monkeypatch.setattr(workload_runner.Shared, "shutdown_managed", lambda: None)
    monkeypatch.setattr(workload_runner, "emit", lambda *_args, **_kwargs: None)
    execute_image_job(
        path, plan.job_id, benchmark_factory=lambda: Benchmark(),
        ensure_comfyui=lambda selected: selected == comfyui,
    )
    assert export_images(path, plan.job_id)["sdxl"]["resolutions"]["1024x1024"][
        "sec_per_image_mean"
    ] == 2.0


def test_supervised_accuracy_projects_committed_question(monkeypatch, tmp_path):
    questions = [{
        "id": "q1", "category": "general", "choices": {"A": "x", "B": "y"},
        "answer": "B",
    }]
    monkeypatch.setattr(supervised_stage, "selected_questions", lambda _stage, _sample: questions)
    monkeypatch.setattr(Shared, "file_hash", lambda _path: "bank-v1")
    plan = RunPlan.create(
        application_version="6.0-pre7", engine_name="fake", tests=["mcq"],
        stage_order=["mcq"], models={
            "llm": [{"tag": "fake:model", "short": "fake"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "cpu_only": False, "force_all": False,
        },
    )
    path = (tmp_path / "results.events.sqlite3").resolve()

    class Supervisor:
        def __init__(self, _spec): pass

        def run(self, callback):
            stage = AccuracyEventStage(
                path, plan, "mcq", questions, "bank-v1", MCQBenchmark.score,
                lambda _results, _answers: None, initialize=False,
            )
            stage.record_question(
                {"tag": "fake:model", "short": "fake", "label": "Fake"},
                "q1", "B", "The answer is B", "ok",
            )
            stage.finish()
            stage.close()
            callback({"kind": "event"})
            callback({"kind": "terminal", "status": "complete"})
            return 0

        @staticmethod
        def cancel(): pass

    result = run_supervised_stage(plan, path, "mcq", lambda _: None, Supervisor)
    assert result["fake"]["accuracy_pct"] == 100.0
    sidecar = json.loads((tmp_path / "answers_mcq_results.json").read_text())
    assert sidecar["fake"]["answers"][0]["raw_response"] == "The answer is B"


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


def test_runner_initializes_runtime_without_cache_configuration_hook():
    from scripts.runtime.workload_runner import configure_runner_engine

    class Engine:
        configured = None

        def runtime_backend(self, hardware_backend, *, cpu_only=False):
            self.configured = (hardware_backend, cpu_only)
            return "cpu" if cpu_only else hardware_backend

    engine = Engine()
    assert configure_runner_engine(engine, "xpu", False) == "auto"
    assert engine.configured == ("xpu", False)


def test_runner_restores_recorded_gpu_split_mode(monkeypatch):
    from scripts.runtime import config
    from scripts.runtime.workload_runner import apply_runner_settings

    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "layer")
    monkeypatch.setattr(config, "RUN_TIMEOUT", 300)
    apply_runner_settings({"gpu_split_mode": "tensor", "run_timeout_seconds": 7})
    assert config.LLAMACPP_GPU_SPLIT_MODE == "tensor"
    assert config.RUN_TIMEOUT == 7


def test_legacy_runner_plan_defaults_gpu_split_to_layer(monkeypatch):
    from scripts.runtime import config
    from scripts.runtime.workload_runner import apply_runner_settings

    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "tensor")
    apply_runner_settings({})
    assert config.LLAMACPP_GPU_SPLIT_MODE == "layer"
