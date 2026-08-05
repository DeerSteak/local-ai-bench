"""Standalone Tk progress window for the unattended setup phase."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


TERMINAL_STATUSES = {"complete", "action_items", "stopped"}


def read_progress_status(path: Path) -> str:
    try:
        status = json.loads(path.read_text()).get("status", "running")
    except (OSError, ValueError, AttributeError):
        return "running"
    return status if status in TERMINAL_STATUSES else "running"


def progress_status_text(status: str) -> str:
    return {
        "complete": "Setup completed successfully.",
        "action_items": "Setup finished with action items. Review the terminal for details.",
        "stopped": "Setup stopped. Review the terminal for details.",
    }.get(status, "Downloading and installing components…")


def start_setup_progress() -> tuple[subprocess.Popen, Path]:
    handle, raw_path = tempfile.mkstemp(prefix="local-ai-bench-setup-", suffix=".json")
    os.close(handle)
    path = Path(raw_path)
    path.write_text(json.dumps({"status": "running"}))
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "scripts.setup.setup_progress",
            "--status-file", str(path), "--parent-pid", str(os.getpid()),
        ])
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return process, path


def finish_setup_progress(path: Path, status: str) -> None:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown setup progress status: {status}")
    if path.exists():
        path.write_text(json.dumps({"status": status}))


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_progress_window(status_file: Path, parent_pid: int) -> None:  # pragma: no cover
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.withdraw()
    root.title("Local AI Bench Setup")
    root.geometry("560x210")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Installing Local AI Bench", font=("TkDefaultFont", 17, "bold")).pack(anchor="w")
    status_label = ttk.Label(frame, text=progress_status_text("running"), wraplength=510)
    status_label.pack(anchor="w", pady=(14, 12))
    progress = ttk.Progressbar(frame, mode="indeterminate", length=500)
    progress.pack(fill="x")
    progress.start(12)
    def close_window() -> None:
        status_file.unlink(missing_ok=True)
        root.destroy()

    close_button = ttk.Button(frame, text="Close", command=close_window, state="disabled")
    close_button.pack(anchor="e", pady=(18, 0))

    def bring_to_front() -> None:
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.after(600, lambda: root.attributes("-topmost", False))

    def poll() -> None:
        status = read_progress_status(status_file)
        if status == "running" and not process_is_running(parent_pid):
            status = "stopped"
        if status in TERMINAL_STATUSES:
            progress.stop()
            progress.configure(mode="determinate", value=100)
            status_label.configure(text=progress_status_text(status))
            close_button.configure(state="normal")
            status_file.unlink(missing_ok=True)
            root.lift()
            return
        root.after(400, poll)

    root.after(100, bring_to_front)
    root.after(400, poll)
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    run_progress_window(args.status_file, args.parent_pid)


if __name__ == "__main__":  # pragma: no cover
    main()
