"""Shared llama.cpp tool discovery for setup and benchmark execution."""

import platform
import shutil
from pathlib import Path

from scripts.runtime import config
from scripts.setup.setup_config import configured_llamacpp_tool, load_setup_config


# The WSL-Ubuntu CUDA toolkit installs here and never puts itself on PATH.
CUDA_BIN_DIRS = ("/usr/local/cuda/bin",)


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
    if platform_name == "Darwin" and vendored_dir.exists():
        managed = next((path for path in vendored_dir.rglob(exe_name) if path.is_file()), None)
        if managed is not None:
            return str(managed)
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
