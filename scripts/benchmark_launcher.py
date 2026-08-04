#!/usr/bin/env python3
"""Select the graphical or terminal benchmark frontend."""

import argparse
import os
import platform
import sys

from interface_mode import desktop_available, select_interface_mode


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def main():  # pragma: no cover — frontend dispatch entrypoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", choices=("auto", "gui", "terminal"), default="auto")
    args = parser.parse_args()
    gui_available = tkinter_available()
    try:
        mode = select_interface_mode(
            args.interface, platform_name=platform.system(), env=dict(os.environ),
            stdin_is_tty=sys.stdin.isatty(), gui_available=gui_available,
        )
    except ValueError as exc:
        print(f"Cannot start benchmark interface: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if mode == "gui":
        from benchmark_gui import run_benchmark_gui
        raise SystemExit(run_benchmark_gui())
    if mode == "terminal":
        from benchmark_frontend import run_frontend
        raise SystemExit(run_frontend())
    hint = "Pass benchmark options to run noninteractively."
    if not desktop_available(platform.system(), dict(os.environ)):
        hint = "No desktop or interactive terminal was detected. " + hint
    print(hint, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
