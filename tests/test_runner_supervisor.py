import signal
import subprocess
from pathlib import Path
from typing import cast

import pytest

from scripts.runtime import config
from scripts.runtime.engines.base import GenerationMeasurement
from scripts.results.llm_event_stage import LLMEventStage, export_llm_section
from scripts.results.native_bench_event_stage import NativeBenchEventStage, export_native_bench_section
from scripts.results.run_plan import RunPlan
from scripts.runtime import runner_supervisor
from scripts.runtime import workload_runner
from scripts.runtime.runner_supervisor import (
    RUNNER_EVENT_PREFIX, RunnerHeartbeatTimeout, RunnerSpec, RunnerSupervisor, SupervisedProcess,
    build_runner_command, parse_runner_event,
)


def spec(tmp_path):
    return RunnerSpec("job_abc", "llm", (tmp_path / "events.sqlite3").resolve())


def test_runner_command_is_fixed_and_contains_no_caller_command_surface(tmp_path):
    command = build_runner_command(spec(tmp_path), "/venv/python")
    assert command == [
        "/venv/python", "-m", "scripts.runtime.workload_runner",
        "--job-id", "job_abc", "--stage", "llm", "--event-store",
        str((tmp_path / "events.sqlite3").resolve()),
    ]
    assert not any(token in command for token in ("-c", "--command", "--env", "--executable"))


@pytest.mark.parametrize("value", [
    RunnerSpec("bad", "llm", Path("/tmp/events")),
    RunnerSpec("job_x", "img", Path("/tmp/events")),
    RunnerSpec("job_x", "llm", Path("relative")),
])
def test_runner_spec_rejects_unowned_or_unsupported_execution(value):
    with pytest.raises(ValueError):
        value.validate()


def test_runner_event_requires_matching_ownership_token_and_shape():
    line = RUNNER_EVENT_PREFIX + (
        '{"ownership_token":"token","kind":"heartbeat","timestamp":1.5}'
    )
    heartbeat_event = parse_runner_event(line, "token")
    assert heartbeat_event is not None and heartbeat_event["kind"] == "heartbeat"
    assert parse_runner_event(line, "other") is None
    assert parse_runner_event(RUNNER_EVENT_PREFIX + "{bad", "token") is None
    assert parse_runner_event("ordinary output", "token") is None
    sequenced_event = parse_runner_event(
        RUNNER_EVENT_PREFIX
        + '{"ownership_token":"token","kind":"event","timestamp":1,"sequence":1,"event":{}}',
        "token",
    )
    assert sequenced_event is not None and sequenced_event["sequence"] == 1
    assert parse_runner_event(
        RUNNER_EVENT_PREFIX
        + '{"ownership_token":"token","kind":"event","timestamp":1,"sequence":true,"event":{}}',
        "token",
    ) is None


def test_supervisor_start_owns_process_group_and_private_token(tmp_path):
    captured = {}

    class Process:
        stdout = []

    def factory(command, **options):
        captured.update(command=command, options=options)
        return Process()

    supervisor = RunnerSupervisor(
        spec(tmp_path), process_factory=cast("type[subprocess.Popen]", factory), system="Linux",
    )
    supervisor.start()
    assert captured["options"]["start_new_session"] is True
    assert captured["options"]["env"]["LOCAL_AI_BENCH_RUNNER_TOKEN"] == supervisor.ownership_token
    assert supervisor.ownership_token not in captured["command"]


def test_heartbeat_uses_supervisor_receive_time_and_times_out(tmp_path):
    now = [10.0]

    class Process:
        @staticmethod
        def poll():
            return None

    supervisor = RunnerSupervisor(spec(tmp_path), heartbeat_timeout=5, clock=lambda: now[0])
    supervisor.process = cast(SupervisedProcess, Process())
    supervisor.last_heartbeat = now[0]
    events = []
    supervisor.accept_line(
        RUNNER_EVENT_PREFIX
        + f'{{"ownership_token":"{supervisor.ownership_token}","kind":"heartbeat","timestamp":0}}',
        events.append,
    )
    now[0] = 15.1
    with pytest.raises(RunnerHeartbeatTimeout, match="5 seconds"):
        supervisor.check_heartbeat()


def test_unstructured_or_wrong_owner_output_is_only_a_log(tmp_path):
    supervisor = RunnerSupervisor(spec(tmp_path))
    events = []
    supervisor.accept_line("model output\n", events.append)
    supervisor.accept_line(
        RUNNER_EVENT_PREFIX + '{"ownership_token":"wrong","kind":"terminal","timestamp":1}',
        events.append,
    )
    assert events == [
        {"kind": "log", "text": "model output\n"},
        {"kind": "log", "text": RUNNER_EVENT_PREFIX
         + '{"ownership_token":"wrong","kind":"terminal","timestamp":1}'},
    ]


def test_unix_cancel_escalates_only_owned_process_group(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

        @staticmethod
        def wait(timeout=None):
            calls.append(("wait", timeout))
            if len([call for call in calls if call[0] == "wait"]) < 3:
                raise subprocess.TimeoutExpired("runner", timeout or 0)

        @staticmethod
        def terminate():
            calls.append(("terminate",))

        @staticmethod
        def kill():
            calls.append(("kill",))

    monkeypatch.setattr(runner_supervisor.os, "getpgid", lambda pid: pid + 1000)
    monkeypatch.setattr(runner_supervisor.os, "killpg", lambda group, sig: calls.append(
        ("signal", group, sig)))
    supervisor = RunnerSupervisor(spec(tmp_path), graceful_timeout=2, system="Linux")
    supervisor.process = cast(SupervisedProcess, Process())
    supervisor.cancel()
    assert calls == [
        ("signal", 1123, signal.SIGINT), ("wait", 2), ("terminate",),
        ("wait", 2), ("kill",), ("wait", None),
    ]


def test_internal_runner_requires_ownership_token(monkeypatch, capsys):
    monkeypatch.delenv("LOCAL_AI_BENCH_RUNNER_TOKEN", raising=False)
    assert workload_runner.main([
        "--job-id", "job_x", "--stage", "llm", "--event-store", "/tmp/events",
    ]) == 2
    assert "ownership token is required" in capsys.readouterr().err.lower()


def test_internal_runner_executes_journal_plan_and_emits_commit(monkeypatch, tmp_path, capsys):
    path = tmp_path / "events.sqlite3"
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["llm"], stage_order=["llm"],
        models={
            "llm": [{"tag": "fake:model", "short": "fake"}],
            "concurrency": [], "embeddings": [], "images": [],
        },
        effective_config={
            "runs": 1, "warmup_runs": 0, "run_timeout_seconds": 7,
            "accuracy_timeout_seconds": 60, "accuracy_token_budget": 256,
            "cpu_only": True, "force_all": False, "max_prompt_tokens": None,
            "context_lengths": [512], "llamabench_pp": [512],
            "llamabench_tg": [128], "sample_size": None,
            "concurrency_tool_levels": [1, 2], "concurrency_chat_levels": [1, 2],
            "concurrency_tool_context": 512, "concurrency_chat_context": 1024,
            "concurrency_chat_soft_exit_floor": 2,
        },
    )
    stage = LLMEventStage(path, plan, lambda _: None)
    stage.close()

    class Engine:
        def start(self, *, gpu_visible):
            assert gpu_visible is False
            return True

        @staticmethod
        def available():
            return False

    class Benchmark:
        def run(self, **kwargs):
            assert kwargs["context_lengths"] == [512]
            assert kwargs["models"][0]["label"] == "fake:model"
            sample = GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)
            kwargs["journal"].record_case(kwargs["models"][0], 512, "512", [sample], "ok", 1)
            kwargs["journal"].finish()

    monkeypatch.setenv("LOCAL_AI_BENCH_RUNNER_TOKEN", "token")
    execute_llm_job = workload_runner.execute_llm_job
    monkeypatch.setattr(workload_runner, "execute_llm_job", lambda path, job_id:
                        execute_llm_job(
                            path, job_id, engine_factory=lambda _: Engine(),
                            benchmark_factory=Benchmark,
                        ))
    old_runs, old_timeout = config.N_RUNS, config.RUN_TIMEOUT
    try:
        assert workload_runner.main([
            "--job-id", plan.job_id, "--stage", "llm", "--event-store", str(path),
        ]) == 0
    finally:
        config.N_RUNS, config.RUN_TIMEOUT = old_runs, old_timeout
    output = capsys.readouterr().out
    assert '"ownership_token":"token"' in output
    assert '"kind":"event"' in output
    assert '"status":"complete"' in output
    assert export_llm_section(path, plan.job_id)["fake"]["512"]["tps_mean"] == 50


def test_conversation_runner_uses_llm_preflight_and_commits_projection(
        tmp_path, monkeypatch, capsys):
    path = tmp_path / "events.sqlite3"
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["llm", "conv"],
        stage_order=["llm", "conv"], models={
            "llm": [
                {"tag": "fake:model", "short": "fake"},
                {"tag": "slow:model", "short": "slow"},
            ],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "run_timeout_seconds": 7,
            "accuracy_timeout_seconds": 60, "accuracy_token_budget": 256,
            "cpu_only": False, "force_all": False, "max_prompt_tokens": 2048,
            "context_lengths": [512], "llamabench_pp": [512],
            "llamabench_tg": [128], "sample_size": None,
            "concurrency_tool_levels": [1, 2], "concurrency_chat_levels": [1, 2],
            "concurrency_tool_context": 512, "concurrency_chat_context": 1024,
            "concurrency_chat_soft_exit_floor": 2,
        },
    )
    model = {"tag": "fake:model", "short": "fake", "label": "Fake"}
    llm = LLMEventStage(path, plan, lambda _: None)
    llm.record_case(
        model, 512, "0.5K", [GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)], "ok", 1,
    )
    llm.record_case(
        {"tag": "slow:model", "short": "slow", "label": "Slow"}, 512, "0.5K",
        [GenerationMeasurement(0.2, 100, 1, 100.2, 100.0)], "ok", 1,
    )
    llm.finish()
    llm.close()
    conversation = LLMEventStage(path, plan, lambda _: None, stage_name="conv")
    conversation.close()

    class Engine:
        name = "fake"

        @staticmethod
        def start(*, gpu_visible):
            return gpu_visible

        @staticmethod
        def available():
            return False

    class Benchmark:
        def run(self, **kwargs):
            assert [entry["tag"] for entry in kwargs["models"]] == ["fake:model"]
            assert kwargs["max_prompt_tokens"] == 2048
            kwargs["journal"].record_case(
                kwargs["models"][0], 0, "0K",
                [GenerationMeasurement(0.1, 96, 48, 2.1, 2.0)], "ok", 1,
                depth_tokens=400,
            )
            kwargs["journal"].finish()

    old_timeout = config.RUN_TIMEOUT
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    workload_runner.set_progress_engine("fake")
    try:
        workload_runner.execute_conversation_job(
            path, plan.job_id, engine_factory=lambda _: Engine(), benchmark_factory=Benchmark,
        )
    finally:
        config.RUN_TIMEOUT = old_timeout
        workload_runner.set_progress_engine(None)
    result = export_llm_section(path, plan.job_id, "conv")
    assert result["fake"]["0K"]["depth_tokens"] == 400
    assert result["slow"]["skip_reason"] == "slow_tps"
    progress = capsys.readouterr().out
    assert '"stage":"conv","status":"skipped","engine":"fake","model":"slow:model"' in progress


def test_native_runner_reconstructs_plan_and_streams_rows_to_journal(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["llamabench"],
        stage_order=["llamabench"], models={
            "llm": [{"tag": "fake:model", "short": "fake"}],
            "concurrency": [], "embeddings": [], "images": [],
        }, effective_config={
            "runs": 2, "warmup_runs": 0, "run_timeout_seconds": 7,
            "accuracy_timeout_seconds": 60, "accuracy_token_budget": 256,
            "cpu_only": False, "force_all": False, "max_prompt_tokens": None,
            "context_lengths": [512], "llamabench_pp": [512],
            "llamabench_tg": [128], "sample_size": None,
            "concurrency_tool_levels": [1, 2], "concurrency_chat_levels": [1, 2],
            "concurrency_tool_context": 512, "concurrency_chat_context": 1024,
            "concurrency_chat_soft_exit_floor": 2,
        },
    )
    owner = NativeBenchEventStage(path, plan, lambda _: None)
    owner.close()

    class Benchmark:
        def run(self, **kwargs):
            assert kwargs["reps"] == 2
            assert config.LLAMABENCH_PP == [512]
            model = kwargs["models"][0]
            kwargs["journal"].record_model_plan(model, 1, 2)
            kwargs["journal"].record_entry(model, {
                "n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 100.0,
                "samples_ts": [99.0, 101.0], "ts_runs": [99.0, 101.0],
                "requested_reps": 2, "completed_reps": 2,
            })
            kwargs["journal"].finish()

    old_pp, old_tg = config.LLAMABENCH_PP, config.LLAMABENCH_TG
    try:
        workload_runner.execute_llamabench_job(
            path, plan.job_id, engine_factory=lambda _: object(),
            benchmark_factory=Benchmark,
        )
    finally:
        config.LLAMABENCH_PP, config.LLAMABENCH_TG = old_pp, old_tg
    result = export_native_bench_section(path, plan.job_id)["fake"]
    assert result["completed_cases"] == 1
    assert result["prefill_entries"][0]["avg_ts"] == 100


def test_concurrency_runner_uses_plan_shape_and_commits_final_batch(tmp_path):
    path = tmp_path / "events.sqlite3"
    plan = RunPlan.create(
        application_version="4.1", engine_name="fake", tests=["conc_chat"],
        stage_order=["conc_chat"], models={
            "llm": [], "concurrency": [{"tag": "fake:model", "short": "fake"}],
            "embeddings": [], "images": [],
        }, effective_config={
            "runs": 1, "warmup_runs": 0, "run_timeout_seconds": 7,
            "accuracy_timeout_seconds": 60, "accuracy_token_budget": 256,
            "cpu_only": False, "force_all": False, "max_prompt_tokens": None,
            "context_lengths": [512], "llamabench_pp": [512],
            "llamabench_tg": [128], "sample_size": None,
            "concurrency_tool_levels": [1, 2], "concurrency_chat_levels": [1, 4],
            "concurrency_tool_context": 512, "concurrency_chat_context": 2048,
            "concurrency_chat_soft_exit_floor": 4,
        },
    )
    owner = LLMEventStage(
        path, plan, lambda _: None, stage_name="conc_chat", model_family="concurrency",
    )
    owner.close()

    class Engine:
        name = "fake"

        @staticmethod
        def start(*, gpu_visible):
            return gpu_visible

        @staticmethod
        def available():
            return False

    class Benchmark:
        def run(self, **kwargs):
            assert kwargs["levels"] == [1, 4]
            assert kwargs["per_request_context"] == 2048
            assert kwargs["soft_exit_floor"] == 4
            assert kwargs["stage_name"] == "conc_chat"
            sample = GenerationMeasurement(0.2, 100, 50, 2.2, 2.0)
            kwargs["journal"].record_case(
                kwargs["models"][0], 4, "4", [sample], "ok", 4,
                result_fields={"aggregate_tps": 50.0, "total_tokens": 100,
                               "batch_elapsed_sec": 2.0, "memory": {}},
            )
            kwargs["journal"].finish()

    old_timeout = config.RUN_TIMEOUT
    try:
        workload_runner.execute_concurrency_job(
            path, plan.job_id, "conc_chat", engine_factory=lambda _: Engine(),
            benchmark_factory=Benchmark,
        )
    finally:
        config.RUN_TIMEOUT = old_timeout
    result = export_llm_section(path, plan.job_id, "conc_chat", "concurrency")
    assert result["fake"]["4"]["aggregate_tps"] == 50
