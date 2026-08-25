"""Process launch and exit coordination for the benchmark GUI."""

import os
import platform
import subprocess
from pathlib import Path

from scripts.runtime import config, hardware
from scripts.runtime.pause_control import PAUSE_CONTROL_ENV, create_pause_control
from scripts.runtime.shared import RUN_LOG_UTC_OFFSET_ENV


PROCESS_EXIT_DRAIN_GRACE_SECONDS = 0.25


def should_finalize_process_exit(exit_code: int | None, reader_done: bool,
                                 exit_observed_at: float | None, last_output_at: float,
                                 now: float,
                                 grace: float = PROCESS_EXIT_DRAIN_GRACE_SECONDS) -> bool:
    """Wait for reader completion or a quiet drain period after parent exit."""
    if exit_code is None or exit_observed_at is None:
        return False
    return reader_done or now - max(exit_observed_at, last_output_at) >= grace


def open_path_command(path: Path, system: str) -> list[str]:
    if system == "Darwin":
        return ["open", str(path)]
    if system == "Windows":
        return ["explorer", str(path)]
    return ["xdg-open", str(path)]


def open_path_process_options(system: str) -> dict:
    options = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if system != "Windows":
        options["start_new_session"] = True
    return options


def launch_controlled_process(command: list[str], *, creationflags: int = 0,
                              pause_path_factory=create_pause_control,
                              popen=subprocess.Popen, utc_offset_fn=None,
                              ) -> tuple[subprocess.Popen, Path]:
    control_path = pause_path_factory()
    child_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_COLOR": "1",
        "LOCAL_AI_BENCH_PROGRESS": "1",
        PAUSE_CONTROL_ENV: str(control_path),
    }
    utc_offset = (utc_offset_fn or windows_host_utc_offset_minutes)()
    if utc_offset is not None:
        child_env[RUN_LOG_UTC_OFFSET_ENV] = str(utc_offset)
    try:
        process = popen(
            command, cwd=config.SCRIPT_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=creationflags, env=child_env,
        )
    except BaseException:
        control_path.unlink(missing_ok=True)
        raise
    return process, control_path


def windows_host_utc_offset_minutes(*, system=platform.system, release=platform.release,
                                    run=subprocess.run) -> int | None:
    """Read the Windows host's current UTC offset when the GUI runs under WSL."""
    if not hardware.detect_wsl(system(), release()):
        return None
    try:
        result = run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "[int][TimeZoneInfo]::Local.GetUtcOffset([DateTimeOffset]::UtcNow).TotalMinutes"],
            capture_output=True, text=True, timeout=10,
        )
        minutes = int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return minutes if result.returncode == 0 and -14 * 60 <= minutes <= 14 * 60 else None
