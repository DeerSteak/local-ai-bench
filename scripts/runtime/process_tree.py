"""Owned subprocess-tree shutdown."""

import os
import signal
import subprocess
from typing import Any

import psutil


def stop_process_tree(process: Any, *, interrupt: bool = True,
                      timeout: float = 10, system: str | None = None) -> None:
    """Stop a spawned process and descendants captured before its parent can exit."""
    windows = (system == "Windows") if system is not None else os.name == "nt"
    try:
        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
    except psutil.Error:
        descendants = []
    if process.poll() is None:
        try:
            if windows:
                process.send_signal(
                    getattr(signal, "CTRL_BREAK_EVENT", signal.SIGINT)
                    if interrupt else signal.SIGTERM
                )
            else:
                os.killpg(process.pid, signal.SIGINT if interrupt else signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    alive = []
    for descendant in descendants:
        try:
            descendant.terminate()
            alive.append(descendant)
        except psutil.Error:
            continue
    _, alive = psutil.wait_procs(alive, timeout=timeout)
    for descendant in alive:
        try:
            descendant.kill()
        except psutil.Error:
            pass
    if process.poll() is None:
        process.kill()
        process.wait()
