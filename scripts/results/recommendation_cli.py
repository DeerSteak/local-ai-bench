#!/usr/bin/env python3
"""Create an authoritative recommendation artifact from benchmark results."""

import argparse
import json
from pathlib import Path

from scripts.results.recommendation import evaluate_recommendation
from scripts.results.result_history import load_result
from scripts.results.result_store import atomic_write_json, validate_json_data
from scripts.runtime.shared import Shared


def load_constraints(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    validate_json_data(value)
    if not isinstance(value, dict):
        raise ValueError("recommendation constraints must be an object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a constraint-first Local AI Bench recommendation")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = evaluate_recommendation(
            [load_result(path) for path in args.results], load_constraints(args.constraints))
        atomic_write_json(args.out, artifact)
        Shared.ok(f"Wrote recommendation artifact to {args.out}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        Shared.err(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
