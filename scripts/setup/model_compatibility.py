"""Imported-model architecture inspection and read-only runtime checks."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


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
    if reader_factory is None:
        import gguf
        reader_factory = gguf.GGUFReader
    try:
        field = reader_factory(str(path)).fields.get("general.architecture")
        value = field.contents() if field is not None else None
    except Exception:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) and value else None


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
