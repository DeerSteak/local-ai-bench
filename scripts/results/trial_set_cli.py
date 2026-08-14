#!/usr/bin/env python3
"""Build a noise-aware trial-set artifact from compatible benchmark results."""

import argparse
from pathlib import Path

from scripts.results.result_history import load_result
from scripts.results.result_store import atomic_write_json
from scripts.results.trial_set import build_trial_set
from scripts.runtime.shared import Shared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a Local AI Bench repeated-trial comparison")
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = build_trial_set(
            [load_result(path) for path in args.baseline],
            [load_result(path) for path in args.candidate],
        )
        atomic_write_json(args.out, artifact)
        Shared.ok(f"Wrote repeated-trial comparison to {args.out}")
        return 0
    except (OSError, ValueError) as exc:
        Shared.err(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
