"""Descriptive observer-effect screen for paired telemetry result files."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from scripts.results.result_store import validate_json_data


METRIC_BOUNDS_PCT = {"ttft": 2.0, "throughput": 1.0, "wall": 1.0}
MIN_PAIRS = 20


def percentile(values: Sequence[float], proportion: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def extract_case_metrics(result: dict, section: str, model: str, case: str) -> dict[str, float]:
    validate_json_data(result)
    sample = result.get(section, {}).get(model, {}).get(case)
    if not isinstance(sample, dict):
        raise ValueError(f"result has no {section}/{model}/{case} case")
    ttft = sample.get("client_ttft_mean_sec", sample.get("ttft_mean_sec"))
    throughput = sample.get("tps_mean")
    wall_values = [
        float(item["client_wall_sec"]) for item in sample.get("valid_samples", [])
        if isinstance(item, dict) and isinstance(item.get("client_wall_sec"), (int, float))
    ]
    if not isinstance(ttft, (int, float)) or not isinstance(throughput, (int, float)) or not wall_values:
        raise ValueError(f"{section}/{model}/{case} lacks TTFT, throughput, or valid wall samples")
    values = {"ttft": float(ttft), "throughput": float(throughput),
              "wall": float(statistics.median(wall_values))}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("qualification metrics must be finite and positive")
    return values


def metric_impacts(off: dict[str, float], on: dict[str, float]) -> dict[str, float]:
    return {
        "ttft": (on["ttft"] - off["ttft"]) / off["ttft"] * 100,
        "throughput": (off["throughput"] - on["throughput"]) / off["throughput"] * 100,
        "wall": (on["wall"] - off["wall"]) / off["wall"] * 100,
    }


def analyze_pairs(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"observer screen requires at least {MIN_PAIRS} pairs")
    expected_orders = ["off-on" if index % 2 == 0 else "on-off" for index in range(len(pairs))]
    orders = [pair.get("order") for pair in pairs]
    if orders != expected_orders:
        raise ValueError("pair order must alternate off-on, on-off, starting with off-on")
    impacts = [metric_impacts(pair["off"], pair["on"]) for pair in pairs]
    metrics = {}
    passed = True
    for metric, bound in METRIC_BOUNDS_PCT.items():
        values = [impact[metric] for impact in impacts]
        median = statistics.median(values)
        p90 = percentile(values, 0.90)
        metric_passed = median <= bound + 1e-12 and p90 <= bound * 2 + 1e-12
        metrics[metric] = {
            "median_impact_pct": median,
            "p90_impact_pct": p90,
            "min_impact_pct": min(values),
            "max_impact_pct": max(values),
            "median_bound_pct": bound,
            "p90_bound_pct": bound * 2,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {"pair_count": len(pairs), "metrics": metrics, "passed": passed}


def analyze_manifest(manifest: dict, base_dir: Path) -> dict[str, Any]:
    section = manifest.get("section")
    model = manifest.get("model")
    case = manifest.get("case")
    if not isinstance(section, str) or not section:
        raise ValueError("manifest section, model, and case must be non-empty strings")
    if not isinstance(model, str) or not model:
        raise ValueError("manifest section, model, and case must be non-empty strings")
    if not isinstance(case, str) or not case:
        raise ValueError("manifest section, model, and case must be non-empty strings")
    pairs = []
    for record in manifest.get("pairs", []):
        if not isinstance(record, dict):
            raise ValueError("each pair must be an object")
        pair = {"order": record.get("order")}
        for mode in ("off", "on"):
            path_value = record.get(mode)
            if not isinstance(path_value, str) or not path_value:
                raise ValueError(f"pair {mode} path must be non-empty text")
            path = (base_dir / path_value).resolve()
            result = json.loads(path.read_text(encoding="utf-8"))
            pair[mode] = extract_case_metrics(result, section, model, case)
        pairs.append(pair)
    analysis = analyze_pairs(pairs)
    return {
        "schema_version": 1,
        "platform": manifest.get("platform"),
        "interval_sec": manifest.get("interval_sec"),
        "section": section,
        "model": model,
        "case": case,
        **analysis,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = analyze_manifest(manifest, args.manifest.parent)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
