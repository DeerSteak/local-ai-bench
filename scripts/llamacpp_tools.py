"""Shared llama.cpp tool discovery for setup and benchmark execution."""

import platform
import shutil
from pathlib import Path

import config


def find_llamacpp_tool(base_name: str, *, vendored_dir: Path | None = None,
                       platform_name: str | None = None, which_fn=None) -> str | None:
    platform_name = platform_name or platform.system()
    vendored_dir = Path(vendored_dir) if vendored_dir is not None else config.LLAMACPP_DIR
    which_fn = which_fn or shutil.which
    found = which_fn(base_name)
    if found:
        return found
    exe_name = f"{base_name}.exe" if platform_name == "Windows" else base_name
    if platform_name == "Darwin":
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            candidate = Path(prefix) / exe_name
            if candidate.is_file():
                return str(candidate)
    if vendored_dir.exists():
        match = next((path for path in vendored_dir.rglob(exe_name) if path.is_file()), None)
        if match is not None:
            return str(match)
    return None
