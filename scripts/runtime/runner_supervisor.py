"""Fixed-command workload runner supervision and ownership-aware cleanup."""

import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scripts.runtime import config


RUNNER_EVENT_PREFIX = "::local-ai-bench-runner::"
SUPPORTED_RUNNER_STAGES = {"conc_chat", "conc_tool", "conv", "llamabench", "llm"}


class SupervisedProcess(Protocol):
    """What RunnerSupervisor actually calls on a spawned process — a subprocess.Popen
    satisfies this, and so does a narrower test double."""
    stdout: Any
    returncode: int | None
    pid: int

    def poll(self) -> int | None: ...
    def send_signal(self, sig: int) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


class RunnerHeartbeatTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class RunnerSpec:
    job_id: str
    stage: str
    event_store: Path

    def validate(self) -> None:
        if not self.job_id.startswith("job_"):
            raise ValueError("runner requires a valid job identity")
        if self.stage not in SUPPORTED_RUNNER_STAGES:
            raise ValueError(f"unsupported runner stage: {self.stage}")
        if not Path(self.event_store).is_absolute():
            raise ValueError("runner event-store path must be absolute")


def build_runner_command(spec: RunnerSpec, python_executable: str = sys.executable) -> list[str]:
    spec.validate()
    return [
        python_executable, "-m", "scripts.runtime.workload_runner", "--job-id", spec.job_id,
        "--stage", spec.stage, "--event-store", str(spec.event_store),
    ]


def parse_runner_event(line: str, ownership_token: str) -> dict | None:
    if not line.startswith(RUNNER_EVENT_PREFIX):
        return None
    try:
        event = json.loads(line.removeprefix(RUNNER_EVENT_PREFIX))
    except json.JSONDecodeError:
        return None
    if (not isinstance(event, dict) or event.get("ownership_token") != ownership_token
            or event.get("kind") not in {"heartbeat", "event", "terminal"}
            or not isinstance(event.get("timestamp"), (int, float))):
        return None
    required = {
        "heartbeat": {"ownership_token", "kind", "timestamp"},
        "event": {"ownership_token", "kind", "timestamp", "sequence", "event"},
        "terminal": {"ownership_token", "kind", "timestamp", "status", "job_id", "stage"},
    }[event["kind"]]
    if set(event) != required:
        return None
    if event["kind"] == "event" and (
            not isinstance(event["sequence"], int) or isinstance(event["sequence"], bool)
            or event["sequence"] < 1 or not isinstance(event["event"], dict)):
        return None
    if event["kind"] == "terminal" and (
            not isinstance(event["status"], str) or not isinstance(event["job_id"], str)
            or event["stage"] not in SUPPORTED_RUNNER_STAGES):
        return None
    return event


class RunnerSupervisor:
    def __init__(self, spec: RunnerSpec, *, heartbeat_timeout: float = 30,
                 graceful_timeout: float = 10, process_factory=subprocess.Popen,
                 clock=time.monotonic, system: str | None = None):
        spec.validate()
        if heartbeat_timeout <= 0 or graceful_timeout < 0:
            raise ValueError("runner timeouts are invalid")
        self.spec = spec
        self.heartbeat_timeout = heartbeat_timeout
        self.graceful_timeout = graceful_timeout
        self.process_factory = process_factory
        self.clock = clock
        self.system = system or platform.system()
        self.ownership_token = uuid.uuid4().hex
        self.process: SupervisedProcess | None = None
        self.last_heartbeat = None
        self.lines = queue.Queue()

    def start(self) -> SupervisedProcess:
        if self.process is not None:
            raise RuntimeError("runner already started")
        environment = dict(os.environ)
        environment["LOCAL_AI_BENCH_RUNNER_TOKEN"] = self.ownership_token
        environment["PYTHONIOENCODING"] = "utf-8"
        options = {
            "cwd": config.SCRIPT_DIR, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
            "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1,
            "env": environment,
        }
        if self.system == "Windows":
            options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        elif self.system == "Darwin":
            # Keep the controlling terminal so a supervised sampler can reuse sudo's tty ticket.
            options["process_group"] = 0
        else:
            options["start_new_session"] = True
        self.process = self.process_factory(build_runner_command(self.spec), **options)
        self.last_heartbeat = self.clock()
        threading.Thread(target=self._read_output, daemon=True).start()
        return self.process

    def _read_output(self):
        try:
            for line in self.process.stdout if self.process and self.process.stdout else ():
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def accept_line(self, line: str, on_event) -> None:
        event = parse_runner_event(line, self.ownership_token)
        if event is None:
            on_event({"kind": "log", "text": line})
            return
        if event["kind"] == "heartbeat":
            self.last_heartbeat = self.clock()
        on_event(event)

    def check_heartbeat(self) -> None:
        if self.process is not None and self.process.poll() is None \
                and self.clock() - self.last_heartbeat > self.heartbeat_timeout:
            raise RunnerHeartbeatTimeout(
                f"runner heartbeat exceeded {self.heartbeat_timeout:g} seconds"
            )

    def run(self, on_event) -> int | None:
        process = self.start()
        stream_closed = False
        try:
            while process.poll() is None or not stream_closed:
                try:
                    line = self.lines.get(timeout=min(0.25, self.heartbeat_timeout / 4))
                except queue.Empty:
                    self.check_heartbeat()
                    continue
                if line is None:
                    stream_closed = True
                else:
                    self.accept_line(line, on_event)
                self.check_heartbeat()
            return process.returncode
        except BaseException:
            self.cancel()
            raise

    def cancel(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        if self.system == "Windows":
            self.process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT))
        else:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        try:
            self.process.wait(timeout=self.graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            self.process.terminate()
        try:
            self.process.wait(timeout=self.graceful_timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
