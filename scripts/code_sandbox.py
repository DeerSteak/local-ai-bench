"""Restricted child-process boundary for generated Python evaluation."""

import ast
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


MAX_OUTPUT_BYTES = 1024 * 1024
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
BLOCKED_NAMES = {
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec", "getattr",
    "globals", "help", "input", "locals", "open", "setattr", "vars",
}


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    limit_reason: str | None = None


def validate_candidate_code(code):
    """Reject syntax and direct access to imports, introspection, and dunder attributes."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed"
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return f"restricted name is not allowed: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attribute access is not allowed"
    return None


def run_restricted_python(harness, payload, timeout):
    """Run a trusted harness with bounded output and an isolated working directory."""
    with tempfile.TemporaryDirectory(prefix="local-ai-bench-code-") as directory:
        root = Path(directory)
        stdout_path, stderr_path = root / "stdout", root / "stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", harness], stdin=subprocess.PIPE,
                stdout=stdout, stderr=stderr, cwd=root, env={"PYTHONHASHSEED": "0"},
                start_new_session=os.name != "nt", preexec_fn=_resource_limiter(timeout),
            )
            timed_out, limit_reason = _wait_bounded(
                proc, payload, timeout, stdout_path, stderr_path,
            )
        stderr_text = _read_bounded(stderr_path)
        if limit_reason is None and proc.returncode != 0 and "MemoryError" in stderr_text:
            limit_reason = "memory limit"
        return SandboxResult(
            _read_bounded(stdout_path), stderr_text, proc.returncode,
            timed_out, limit_reason,
        )


def sandbox_prelude(candidate_code):
    """Return the trusted in-child policy and restricted candidate namespace."""
    return (
        "import builtins, json, sys\n"
        "try:\n"
        "    import resource\n"
        f"    resource.setrlimit(resource.RLIMIT_DATA, ({MEMORY_LIMIT_BYTES}, {MEMORY_LIMIT_BYTES}))\n"
        "except (ImportError, AttributeError, OSError, ValueError):\n"
        "    pass\n"
        "def _deny(event, args):\n"
        "    blocked = ('open', 'socket.', 'subprocess.', 'ctypes.', 'os.')\n"
        "    if event == 'import' or event.startswith(blocked):\n"
        "        raise PermissionError('sandbox denied ' + event)\n"
        "sys.addaudithook(_deny)\n"
        "_allowed_names = ('__build_class__', 'abs', 'all', 'any', 'bool', 'chr', 'dict', "
        "'enumerate', 'Exception', 'filter', 'float', 'int', 'isinstance', 'len', 'list', "
        "'map', 'max', 'min', 'object', 'ord', 'pow', 'print', 'range', 'reversed', 'round', "
        "'set', 'sorted', 'str', 'sum', 'tuple', 'ValueError', 'TypeError', 'IndexError', "
        "'KeyError', 'RuntimeError', 'zip')\n"
        "_safe_builtins = {name: getattr(builtins, name) for name in _allowed_names}\n"
        "_scope = {'__builtins__': _safe_builtins, '__name__': '__candidate__'}\n"
        f"exec(compile({candidate_code!r}, '<candidate>', 'exec'), _scope, _scope)\n"
    )


def _resource_limiter(timeout):
    if os.name == "nt":
        return None

    def apply_limits():
        import resource
        cpu_soft = max(1, math.ceil(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

    return apply_limits


def _wait_bounded(proc, payload, timeout, stdout_path, stderr_path):
    try:
        proc.stdin.write(payload.encode("utf-8"))
        proc.stdin.close()
    except BrokenPipeError:
        pass
    deadline = time.monotonic() + timeout
    process = psutil.Process(proc.pid)
    reason = None
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            reason = "timeout"
        else:
            try:
                if process.memory_info().rss > MEMORY_LIMIT_BYTES:
                    reason = "memory limit"
            except psutil.Error:
                pass
        if not reason and any(path.stat().st_size >= MAX_OUTPUT_BYTES for path in (stdout_path, stderr_path)):
            reason = "output limit"
        if reason:
            _terminate(proc)
            break
        time.sleep(0.01)
    proc.wait()
    if not reason and any(path.stat().st_size >= MAX_OUTPUT_BYTES for path in (stdout_path, stderr_path)):
        reason = "output limit"
    return reason == "timeout", reason


def _terminate(proc):
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (PermissionError, ProcessLookupError):
            pass
    proc.kill()


def _read_bounded(path):
    with path.open("rb") as stream:
        return stream.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")
