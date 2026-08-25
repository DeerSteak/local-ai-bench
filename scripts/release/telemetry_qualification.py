"""Descriptive observer-effect screen for paired telemetry result files."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from scripts.results.result_store import validate_json_data


METRIC_BOUNDS_PCT = {"ttft": 2.0, "throughput": 1.0, "wall": 1.0}
TTFT_MEDIAN_BOUND_SEC = 0.002
MIN_PAIRS = 20
SUSTAINED_THROUGHPUT_BOUND_PCT = 1.0
SUSTAINED_RETENTION_BOUND_POINTS = 1.0


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


def extract_sustained_metrics(result: dict, model: str) -> dict[str, float]:
    validate_json_data(result)
    sample = result.get("sustained", {}).get(model)
    if not isinstance(sample, dict) or not isinstance(sample.get("series"), list):
        raise ValueError(f"result has no sustained/{model} series")
    durations = []
    tokens = []
    for window in sample["series"]:
        if not isinstance(window, dict):
            raise ValueError("sustained series contains a non-object window")
        duration = window.get("duration_sec")
        count = window.get("tokens")
        if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
                or not isinstance(count, (int, float)) or isinstance(count, bool)
                or not math.isfinite(duration) or not math.isfinite(count)
                or duration <= 0 or count < 0):
            raise ValueError("sustained series contains invalid duration or token count")
        durations.append(float(duration))
        tokens.append(float(count))
    retention = sample.get("analysis", {}).get("retention_ratio")
    if (not durations or not isinstance(retention, (int, float)) or isinstance(retention, bool)
            or not math.isfinite(retention) or retention <= 0):
        raise ValueError(f"sustained/{model} lacks valid throughput or retention")
    throughput = sum(tokens) / sum(durations)
    if throughput <= 0:
        raise ValueError(f"sustained/{model} lacks valid throughput or retention")
    return {
        "throughput": throughput,
        "retention_pct": float(retention) * 100,
    }


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
        if metric == "ttft":
            duration_impacts = [pair["on"]["ttft"] - pair["off"]["ttft"] for pair in pairs]
            median_sec = statistics.median(duration_impacts)
            p90_sec = percentile(duration_impacts, 0.90)
            median_failed = median > bound + 1e-12 and median_sec > TTFT_MEDIAN_BOUND_SEC
            p90_failed = p90 > bound * 2 + 1e-12 and p90_sec > TTFT_MEDIAN_BOUND_SEC * 2
            metric_passed = not median_failed and not p90_failed
            metrics[metric].update({
                "median_impact_sec": median_sec,
                "p90_impact_sec": p90_sec,
                "min_impact_sec": min(duration_impacts),
                "max_impact_sec": max(duration_impacts),
                "median_bound_sec": TTFT_MEDIAN_BOUND_SEC,
                "p90_bound_sec": TTFT_MEDIAN_BOUND_SEC * 2,
                "passed": metric_passed,
            })
        passed = passed and metric_passed
    return {"pair_count": len(pairs), "metrics": metrics, "passed": passed}


def analyze_sustained_pairs(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"observer screen requires at least {MIN_PAIRS} pairs")
    expected_orders = ["off-on" if index % 2 == 0 else "on-off" for index in range(len(pairs))]
    if [pair.get("order") for pair in pairs] != expected_orders:
        raise ValueError("pair order must alternate off-on, on-off, starting with off-on")
    throughput_impacts = [
        (pair["off"]["throughput"] - pair["on"]["throughput"])
        / pair["off"]["throughput"] * 100
        for pair in pairs
    ]
    retention_impacts = [
        pair["off"]["retention_pct"] - pair["on"]["retention_pct"]
        for pair in pairs
    ]
    metrics = {}
    passed = True
    for name, values, bound, unit in (
        ("throughput", throughput_impacts, SUSTAINED_THROUGHPUT_BOUND_PCT, "percent"),
        ("retention", retention_impacts, SUSTAINED_RETENTION_BOUND_POINTS, "percentage_points"),
    ):
        median = statistics.median(values)
        p90 = percentile(values, 0.90)
        metric_passed = median <= bound + 1e-12 and p90 <= bound * 2 + 1e-12
        metrics[name] = {
            "median_impact": median, "p90_impact": p90,
            "min_impact": min(values), "max_impact": max(values),
            "median_bound": bound, "p90_bound": bound * 2,
            "unit": unit, "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {"pair_count": len(pairs), "metrics": metrics, "passed": passed}


def validate_temperature_mode(result: dict, enabled: bool) -> None:
    settings = result.get("run", {}).get("effective_config", {})
    if settings.get("temperature_telemetry") is not enabled:
        raise ValueError("temperature pair does not match its declared off/on mode")
    if settings.get("memory_telemetry") is not True or settings.get("power_telemetry") is not True:
        raise ValueError("temperature qualification requires the combined memory and power baseline")
    if enabled and result.get("preflight", {}).get("temperature", {}).get("available") is not True:
        raise ValueError("temperature-on result has no available temperature source")


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
            if manifest.get("telemetry_mode") == "temperature":
                validate_temperature_mode(result, mode == "on")
            pair[mode] = (
                extract_sustained_metrics(result, model) if section == "sustained"
                else extract_case_metrics(result, section, model, case)
            )
        pairs.append(pair)
    analysis = analyze_sustained_pairs(pairs) if section == "sustained" else analyze_pairs(pairs)
    return {
        "schema_version": 1,
        "platform": manifest.get("platform"),
        "interval_sec": manifest.get("interval_sec"),
        "telemetry_mode": manifest.get("telemetry_mode", "memory"),
        "source": manifest.get("source"),
        "scope": manifest.get("scope"),
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
