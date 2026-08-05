"""Versioned, non-secret setup choices shared by setup and benchmarks."""

import json
import os
import tempfile
from pathlib import Path


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, SCHEMA_VERSION}


def load_setup_config(path: Path) -> dict:
    """Load a valid setup configuration, or return an empty configuration."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) and data.get("schema_version") in SUPPORTED_SCHEMA_VERSIONS else {}


def write_setup_config(path: Path, *, comfyui_dir: Path | None,
                       llamacpp_tools: dict[str, str | None],
                       gpu_devices: list[dict] | None = None) -> None:
    """Atomically write durable setup paths without credentials."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "comfyui": {"program_dir": str(comfyui_dir.resolve()) if comfyui_dir else None},
        "llama_cpp": llamacpp_tools,
        "gpu": {"devices": gpu_devices or []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def configured_comfyui_dir(data: dict) -> str | None:
    value = data.get("comfyui", {}).get("program_dir")
    return value if isinstance(value, str) and value else None


def configured_llamacpp_tool(data: dict, base_name: str) -> str | None:
    value = data.get("llama_cpp", {}).get(base_name)
    return value if isinstance(value, str) and value else None


def configured_gpu_devices(data: dict) -> list[dict]:
    devices = data.get("gpu", {}).get("devices", [])
    if not isinstance(devices, list):
        return []
    return [device for device in devices if isinstance(device, dict)]
