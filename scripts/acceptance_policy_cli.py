#!/usr/bin/env python3
"""Evaluate a benchmark result against an explicit acceptance policy."""

import argparse
import json
from pathlib import Path

from acceptance_policy import evaluate_policy, load_policy
from decision_report import load_result
from shared import Shared


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Local AI Bench acceptance policy")
    parser.add_argument("result", type=Path)
    parser.add_argument("policy", type=Path)
    args = parser.parse_args(argv)
    try:
        evaluation = evaluate_policy(load_result(args.result), load_policy(args.policy))
        Shared.output(json.dumps(evaluation, allow_nan=False, indent=2, sort_keys=True))
        return 0 if evaluation["decision"] == "accepted" else 2
    except (OSError, ValueError, KeyError) as exc:
        Shared.err(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
