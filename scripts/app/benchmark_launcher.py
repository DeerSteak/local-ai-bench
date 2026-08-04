#!/usr/bin/env python3
"""Select the graphical or terminal benchmark frontend."""

import argparse
import os
import platform
import subprocess
import sys

from scripts.runtime import config
from scripts.app.interface_mode import desktop_available, select_interface_mode


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def parse_launcher_request(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", "--interface", dest="ui",
                        choices=("auto", "gui", "terminal", "none"), default="auto")
    args, benchmark_args = parser.parse_known_args(argv)
    if args.ui != "none" and benchmark_args:
        parser.error("benchmark options require '--ui none' or no --ui option")
    if args.ui == "none" and not benchmark_args:
        parser.error("--ui none requires benchmark options, such as '--tests llm'")
    return args.ui, benchmark_args


def main():  # pragma: no cover — frontend dispatch entrypoint
    requested, benchmark_args = parse_launcher_request(sys.argv[1:])
    if requested == "none":
        command = [sys.executable, "-m", "scripts.app.benchmark", *benchmark_args]
        raise SystemExit(subprocess.call(command))
    gui_available = tkinter_available()
    try:
        mode = select_interface_mode(
            requested, platform_name=platform.system(), env=dict(os.environ),
            stdin_is_tty=sys.stdin.isatty(), gui_available=gui_available,
        )
    except ValueError as exc:
        print(f"Cannot start benchmark interface: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if mode == "gui":
        from benchmark_gui import run_benchmark_gui
        raise SystemExit(run_benchmark_gui())
    if mode == "terminal":
        from scripts.app.benchmark_frontend import run_frontend
        raise SystemExit(run_frontend())
    hint = "Pass benchmark options to run noninteractively."
    if not desktop_available(platform.system(), dict(os.environ)):
        hint = "No desktop or interactive terminal was detected. " + hint
    print(hint, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
