#!/usr/bin/env python3
"""Safe setup entrypoint; the side-effecting workflow loads only when invoked."""

import runpy


def main() -> None:  # pragma: no cover - launches the real interactive installer
    runpy.run_module("scripts.setup.setup_workflow", run_name="__main__")


if __name__ == "__main__":  # pragma: no cover
    main()
