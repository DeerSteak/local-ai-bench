#!/usr/bin/env python3
"""Create a versioned quantization tradeoff artifact from one result."""

import argparse
import json
from pathlib import Path

from scripts.results.result_history import load_result
from scripts.results.result_store import atomic_write_json, validate_json_data
from scripts.results.variant_comparison import build_variant_comparison
from scripts.runtime.shared import Shared


def load_quality_verdicts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_json_data(value)
    if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(verdict, str)
            for key, verdict in value.items()):
        raise ValueError("quality verdicts must be a variant-to-verdict object")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a GGUF quantization tradeoff artifact")
    parser.add_argument("result", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--performance-section", default="llm")
    parser.add_argument("--case", required=True)
    parser.add_argument("--accuracy-section", required=True)
    parser.add_argument("--quality-verdicts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact = build_variant_comparison(
            load_result(args.result), base_model=args.base_model,
            reference_variant=args.reference, performance_section=args.performance_section,
            case=args.case, accuracy_section=args.accuracy_section,
            quality_verdicts=load_quality_verdicts(args.quality_verdicts),
        )
        atomic_write_json(args.out, artifact)
        Shared.ok(f"Wrote variant comparison artifact to {args.out}")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        Shared.err(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
