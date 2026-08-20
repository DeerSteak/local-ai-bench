"""Shared llama.cpp tool discovery for setup and benchmark execution."""

import platform
import re
import shutil
import subprocess
from pathlib import Path

from scripts.runtime import config
from scripts.setup.setup_config import configured_llamacpp_tool, load_setup_config


# The WSL-Ubuntu CUDA toolkit installs here and never puts itself on PATH.
CUDA_BIN_DIRS = ("/usr/local/cuda/bin",)
LLAMACPP_TOOL_NAMES = ("llama-server", "llama-bench", "llama-batched-bench")


def llamacpp_backend_from_device_listing(output: str) -> str:
    for line in output.splitlines():
        device = line.strip().lower()
        for pattern, backend in (
            (r"cuda\d*\s*:", "cuda"), (r"(?:rocm|hip)\d*\s*:", "rocm"),
            (r"(?:metal|mtl)\d*\s*:", "metal"),
            (r"(?:sycl|level[- ]?zero)\d*\s*:", "xpu"), (r"vulkan\d*\s*:", "vulkan"),
        ):
            if re.match(pattern, device):
                return backend
    return "cpu"


def probe_llamacpp_backend(binary: str | Path, *, run=None) -> str | None:
    run = run or subprocess.run
    try:
        completed = run(
            [str(binary), "--list-devices"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode:
        return None
    return llamacpp_backend_from_device_listing(f"{completed.stdout}\n{completed.stderr}")


def managed_llamacpp_tools(vendored_dir: Path, platform_name: str) -> dict[str, str]:
    exe_suffix = ".exe" if platform_name == "Windows" else ""
    server_name = f"{LLAMACPP_TOOL_NAMES[0]}{exe_suffix}"
    for server in sorted(Path(vendored_dir).rglob(server_name)):
        tools = {
            name: server.parent / f"{name}{exe_suffix}"
            for name in LLAMACPP_TOOL_NAMES
        }
        if all(path.is_file() for path in tools.values()):
            return {name: str(path) for name, path in tools.items()}
    return {}


def find_nvcc(*, which_fn=shutil.which, exists_fn=None) -> str | None:
    """Path to nvcc, including a toolkit that was installed but never added to PATH."""
    on_path = which_fn("nvcc")
    if on_path:
        return on_path
    exists_fn = exists_fn or (lambda path: Path(path).is_file())
    for directory in CUDA_BIN_DIRS:
        candidate = Path(directory) / "nvcc"
        if exists_fn(candidate):
            return str(candidate)
    return None


def cuda_architecture(compute_cap: str | None) -> str | None:
    """CMAKE_CUDA_ARCHITECTURES value for a reported compute capability, e.g. "8.9" -> "89".
    Preferred over cmake's `native`, whose device probe finds nothing under WSL2."""
    parts = str(compute_cap or "").strip().split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return f"{int(parts[0])}{parts[1]}"


def find_llamacpp_tool(base_name: str, *, vendored_dir: Path | None = None,
                       platform_name: str | None = None, which_fn=None) -> str | None:
    platform_name = platform_name or platform.system()
    vendored_dir = Path(vendored_dir) if vendored_dir is not None else config.LLAMACPP_DIR
    which_fn = which_fn or shutil.which
    exe_name = f"{base_name}.exe" if platform_name == "Windows" else base_name
    managed_tools = managed_llamacpp_tools(vendored_dir, platform_name) \
        if vendored_dir.exists() else {}
    if base_name in managed_tools:
        return managed_tools[base_name]
    found = which_fn(base_name)
    if found:
        return found
    if platform_name == "Darwin":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            candidate = Path(prefix) / exe_name
            if candidate.is_file():
                return str(candidate)
    configured = configured_llamacpp_tool(load_setup_config(config.SETUP_CONFIG_PATH), base_name)
    if configured and Path(configured).is_file():
        return configured
    if vendored_dir.exists():
        match = next((path for path in vendored_dir.rglob(exe_name) if path.is_file()), None)
        if match is not None:
            return str(match)
    return None
