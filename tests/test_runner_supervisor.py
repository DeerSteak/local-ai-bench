import signal
import subprocess
from pathlib import Path

import pytest

import config
from engines.base import GenerationMeasurement
from llm_event_stage import LLMEventStage, export_llm_section
from run_plan import RunPlan
import runner_supervisor
import workload_runner
from runner_supervisor import (
    RUNNER_EVENT_PREFIX, RunnerHeartbeatTimeout, RunnerSpec, RunnerSupervisor,
    build_runner_command, parse_runner_event,
)


def spec(tmp_path):
    return RunnerSpec("job_abc", "llm", (tmp_path / "events.sqlite3").resolve())


def test_runner_command_is_fixed_and_contains_no_caller_command_surface(tmp_path):
    command = build_runner_command(spec(tmp_path), "/venv/python")
    assert command == [
        "/venv/python", str(runner_supervisor.config.SCRIPT_DIR / "scripts" / "workload_runner.py"),
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
    assert parse_runner_event(line, "token")["kind"] == "heartbeat"
    assert parse_runner_event(line, "other") is None
    assert parse_runner_event(RUNNER_EVENT_PREFIX + "{bad", "token") is None
    assert parse_runner_event("ordinary output", "token") is None
    assert parse_runner_event(
        RUNNER_EVENT_PREFIX
        + '{"ownership_token":"token","kind":"event","timestamp":1,"sequence":1,"event":{}}',
        "token",
    )["sequence"] == 1
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

    supervisor = RunnerSupervisor(spec(tmp_path), process_factory=factory, system="Linux")
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
    supervisor.process = Process()
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
                raise subprocess.TimeoutExpired("runner", timeout)

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
    supervisor.process = Process()
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
            "cpu_only": True, "force_all": False, "context_lengths": [512],
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
