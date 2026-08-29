"""Versioned, non-secret setup choices shared by setup and benchmarks."""

import json
import os
import tempfile
from pathlib import Path


# 4 adds the optional Vulkan llama.cpp toolset; older files simply lack it.
SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3, SCHEMA_VERSION}


def load_setup_config(path: Path) -> dict:
    """Load a valid setup configuration, or return an empty configuration."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) and data.get("schema_version") in SUPPORTED_SCHEMA_VERSIONS else {}


def write_setup_config(path: Path, *, comfyui_dir: Path | None,
                       llamacpp_tools: dict[str, str | None],
                       llamacpp_vulkan_tools: dict[str, str | None] | None = None,
                       gpu_devices: list[dict] | None = None,
                       vllm: dict | None = None) -> None:
    """Atomically write durable setup paths without credentials."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "comfyui": {"program_dir": str(comfyui_dir.resolve()) if comfyui_dir else None},
        "llama_cpp": llamacpp_tools,
        "llama_cpp_vulkan": llamacpp_vulkan_tools or {},
        "gpu": {"devices": gpu_devices or []},
        "vllm": vllm or {},
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


def vllm_setup_config(*, executable: str | None, launcher: str | None,
                      server_url: str | None, launcher_extra_args: list[str],
                      hf_home: Path | str) -> dict:
    """Build the vLLM handoff consumed by both setup and benchmark engines."""
    return {
        "executable": executable,
        "launcher": launcher,
        "server_url": server_url,
        "launcher_extra_args": list(launcher_extra_args),
        "hf_home": str(hf_home),
    }


def configured_comfyui_dir(data: dict) -> str | None:
    value = data.get("comfyui", {}).get("program_dir")
    return value if isinstance(value, str) and value else None


def configured_llamacpp_tool(data: dict, base_name: str) -> str | None:
    value = data.get("llama_cpp", {}).get(base_name)
    return value if isinstance(value, str) and value else None


def configured_llamacpp_vulkan_tool(data: dict, base_name: str) -> str | None:
    value = data.get("llama_cpp_vulkan", {}).get(base_name)
    return value if isinstance(value, str) and value else None


def configured_vllm(data: dict) -> dict:
    """Recorded vLLM runtime: executable, launcher, server URL, and launcher extra args."""
    value = data.get("vllm")
    return value if isinstance(value, dict) else {}


def configured_vllm_path(data: dict, key: str) -> str | None:
    """One recorded vLLM path — `executable`, `launcher`, or `server_url`."""
    value = configured_vllm(data).get(key)
    return value if isinstance(value, str) and value else None


def configured_vllm_launcher_args(data: dict) -> list[str]:
    """Extra args the platform launcher injects into every run."""
    value = configured_vllm(data).get("launcher_extra_args")
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def configured_gpu_devices(data: dict) -> list[dict]:
    devices = data.get("gpu", {}).get("devices", [])
    if not isinstance(devices, list):
        return []
    return [device for device in devices if isinstance(device, dict)]


def available_gpu_split_modes(data: dict, runtime_backend: str) -> tuple[str, ...]:
    """Return split modes supported by the recorded runtime topology."""
    devices = configured_gpu_devices(data)
    matching = [device for device in devices if device.get("backend") == runtime_backend]
    if runtime_backend in {"cuda", "rocm"} and len(matching) >= 2:
        return "single", "layer", "tensor"
    if runtime_backend in {"cuda", "rocm", "vulkan", "xpu"} and matching:
        return "single", "layer"
    return ("layer",)
