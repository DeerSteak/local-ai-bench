"""Imported-model architecture inspection and read-only runtime checks."""

import json
import socket
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


VLLM_ARCH_PROBE = (
    "import sys; from vllm import ModelRegistry; "
    "print('supported' if sys.argv[1] in ModelRegistry.get_supported_archs() else 'unsupported')"
)


@dataclass(frozen=True)
class ModelCompatibility:
    engine: str
    tag: str
    architecture: str | None
    status: str
    detail: str
    checks: tuple["CompatibilityCheck", ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine, "tag": self.tag, "architecture": self.architecture,
            "status": self.status, "detail": self.detail,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class CompatibilityCheck:
    name: str
    status: str
    severity: str
    detail: str
    scope: str = "model"
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "status": self.status, "severity": self.severity,
            "scope": self.scope, "detail": self.detail, "evidence": self.evidence,
        }


def gguf_metadata(path: Path, reader_factory=None) -> tuple[dict, str | None]:
    if reader_factory is None:
        import gguf
        reader_factory = gguf.GGUFReader
    try:
        fields = reader_factory(str(path)).fields
        values = {key: field.contents() for key, field in fields.items()}
    except Exception as exc:
        return {}, str(exc)
    return values, None


def _text_value(value) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) and value.strip() else None


def chat_template_check(metadata: dict, read_error: str | None = None) -> CompatibilityCheck:
    if read_error:
        return CompatibilityCheck(
            "chat_template", "unreadable", "warning",
            f"GGUF metadata could not be read: {read_error}",
        )
    template = _text_value(metadata.get("tokenizer.chat_template"))
    if template:
        return CompatibilityCheck(
            "chat_template", "passed", "info", "Embedded chat template is present.",
            evidence={"source": "tokenizer.chat_template"},
        )
    return CompatibilityCheck(
        "chat_template", "absent", "warning",
        "No embedded chat template; the runtime may use heuristic formatting.",
        evidence={"source": "runtime_fallback"},
    )


def declared_context_length(metadata: dict) -> int | None:
    for key, value in metadata.items():
        if key.count(".") == 1 and key.endswith(".context_length") \
                and isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def context_capacity_check(declared: int | None, requested: int | None,
                           engine_max: int | None) -> CompatibilityCheck:
    evidence = {"declared": declared, "requested": requested, "engine_max": engine_max}
    if declared is None:
        return CompatibilityCheck(
            "context_capacity", "unknown", "warning",
            "The model's declared context length could not be read.", evidence=evidence,
        )
    limits = [value for value in (declared, engine_max) if isinstance(value, int) and value > 0]
    if requested is not None and limits and requested > min(limits):
        return CompatibilityCheck(
            "context_capacity", "exceeded", "warning",
            f"Requested context {requested} exceeds the readable limit {min(limits)}.",
            evidence=evidence,
        )
    return CompatibilityCheck(
        "context_capacity", "passed", "info", "Requested context is within readable limits.",
        evidence=evidence,
    )


def tool_support_check(supported: bool, tool_selected: bool) -> CompatibilityCheck:
    if not tool_selected:
        return CompatibilityCheck(
            "tool_calls", "not_applicable", "info", "No tool-call workload is selected.",
            scope="tool",
        )
    if supported:
        return CompatibilityCheck(
            "tool_calls", "passed", "info", "Engine tool-call support is configured.",
            scope="tool",
        )
    return CompatibilityCheck(
        "tool_calls", "unsupported", "workload_blocking",
        "Tool-call support is not configured for this model and engine.", scope="tool",
    )


def weight_completeness_check(paths: list[Path] | tuple[Path, ...]) -> CompatibilityCheck:
    paths = [Path(path) for path in paths]
    missing = [str(path) for path in paths if not path.is_file()]
    empty = [str(path) for path in paths if path.is_file() and path.stat().st_size == 0]
    if not paths or missing or empty:
        return CompatibilityCheck(
            "weights", "incomplete", "hard_failure",
            "One or more declared model weight files are missing or empty.",
            evidence={"files": len(paths), "missing": missing, "empty": empty},
        )
    return CompatibilityCheck(
        "weights", "passed", "info", "All declared model weight files are present and non-empty.",
        evidence={"files": len(paths), "bytes": sum(path.stat().st_size for path in paths)},
    )


def preflight_verdict(checks: tuple[CompatibilityCheck, ...], force_all: bool = False) -> str:
    if any(check.severity == "hard_failure" for check in checks):
        return "excluded"
    if any(check.severity == "workload_blocking" for check in checks):
        return "workload_limited"
    if any(check.severity == "warning" for check in checks) and not force_all:
        return "warning"
    return "passed"


def architecture_from_config(path: Path) -> str | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    architectures = payload.get("architectures") if isinstance(payload, dict) else None
    if not isinstance(architectures, list):
        return None
    return next((value for value in architectures if isinstance(value, str) and value), None)


def architecture_from_gguf(path: Path, reader_factory=None) -> str | None:
    metadata, _error = gguf_metadata(path, reader_factory)
    return _text_value(metadata.get("general.architecture"))


def probe_vllm_architecture(python_exe: Path | None, architecture: str | None,
                            *, run=subprocess.run) -> tuple[str, str]:
    if not architecture:
        return "unknown", "Model config does not declare an architecture."
    if python_exe is None:
        return "unavailable", "The vLLM Python interpreter was not found."
    try:
        result = run(
            [str(python_exe), "-c", VLLM_ARCH_PROBE, architecture],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return "unavailable", str(exc)
    verdict = result.stdout.strip().lower()
    if result.returncode == 0 and verdict in {"supported", "unsupported"}:
        detail = ("Architecture is registered by this vLLM installation."
                  if verdict == "supported" else
                  "Architecture is not registered by this vLLM installation.")
        return verdict, detail
    return "unavailable", (result.stderr or result.stdout or "Architecture probe failed").strip()


def reserve_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def llamacpp_probe_command(binary: str, model_path: Path, port: int) -> list[str]:
    return [
        binary, "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port),
        "-ngl", "0", "-c", "512", "-b", "128", "--parallel", "1",
    ]


def probe_llamacpp_load(tag: str, model_path: Path, binary: str | None, *, timeout=180.0,
                        popen: Callable = subprocess.Popen, open_fn=urllib.request.urlopen,
                        monotonic=time.monotonic, sleep=time.sleep,
                        port_factory=reserve_loopback_port, control=None) -> ModelCompatibility:
    architecture = architecture_from_gguf(model_path)
    if binary is None:
        return ModelCompatibility(
            "llamacpp", tag, architecture, "unavailable", "llama-server was not found.",
        )
    port = port_factory()
    with tempfile.NamedTemporaryFile(mode="w+", suffix="-llamacpp-probe.log") as log:
        try:
            process = popen(
                llamacpp_probe_command(binary, model_path, port),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        except OSError as exc:
            return ModelCompatibility("llamacpp", tag, architecture, "unavailable", str(exc))
        if control is not None:
            control.track_process(process)
        deadline = monotonic() + timeout
        try:
            while monotonic() < deadline:
                if control is not None and control.cancelled:
                    return ModelCompatibility(
                        "llamacpp", tag, architecture, "cancelled", "Model-load probe cancelled.",
                    )
                if process.poll() is not None:
                    log.seek(0)
                    detail = log.read().strip()[-2000:] or "llama-server exited during model load."
                    return ModelCompatibility("llamacpp", tag, architecture, "load_failed", detail)
                try:
                    with open_fn(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if 200 <= getattr(response, "status", 200) < 300:
                            return ModelCompatibility(
                                "llamacpp", tag, architecture, "supported",
                                "Model loaded successfully CPU-only; GPU compatibility was not tested.",
                            )
                except Exception:
                    pass
                sleep(0.2)
            return ModelCompatibility(
                "llamacpp", tag, architecture, "timed_out",
                f"Model did not finish loading within {timeout:g} seconds.",
            )
        finally:
            if control is not None:
                control.clear_process(process)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def inspect_llamacpp_model(tag: str, model_path: Path) -> ModelCompatibility:
    architecture = architecture_from_gguf(model_path)
    detail = ("Architecture metadata identified; llama.cpp requires a bounded model-load probe."
              if architecture else "GGUF architecture metadata could not be read.")
    return ModelCompatibility(
        "llamacpp", tag, architecture, "load_probe_required" if architecture else "unknown", detail,
    )


def inspect_vllm_model(tag: str, config_path: Path, python_exe: Path | None,
                       *, run=subprocess.run) -> ModelCompatibility:
    architecture = architecture_from_config(config_path)
    status, detail = probe_vllm_architecture(python_exe, architecture, run=run)
    return ModelCompatibility("vllm", tag, architecture, status, detail)
