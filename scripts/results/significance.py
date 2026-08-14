"""Noise-aware comparison primitives for single runs and independent trials."""

import math

from scripts.runtime import config


THROUGHPUT_METRICS = {"tps_mean", "aggregate_tps", "chunks_per_sec_mean"}
LATENCY_METRICS = {"ttft_mean_sec", "client_ttft_mean_sec", "server_prompt_mean_sec"}
WALL_TIME_METRICS = {"sec_per_image_mean"}


def practical_threshold_pct(metric: str) -> float:
    """Return the predeclared provisional practical-change floor."""
    if metric in THROUGHPUT_METRICS:
        return config.PRACTICAL_THROUGHPUT_THRESHOLD_PCT
    if metric in LATENCY_METRICS:
        return config.PRACTICAL_TTFT_THRESHOLD_PCT
    if metric in WALL_TIME_METRICS:
        return config.PRACTICAL_WALL_TIME_THRESHOLD_PCT
    return config.PRACTICAL_ACCURACY_THRESHOLD_PCT


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric_evidence(values: dict, metric: str, dispersion_key: str | None) -> dict:
    """Normalize one aggregate without treating absent dispersion as zero."""
    value = finite_number(values.get(metric))
    dispersion = finite_number(values.get(dispersion_key)) if dispersion_key else None
    count = values.get("valid_runs", values.get("n_runs"))
    if metric == "accuracy_pct":
        count = values.get("total", count)
    sample_count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
    return {"value": value, "dispersion": dispersion, "sample_count": sample_count}


def compare_metric(metric: str, baseline: dict | None, candidate: dict | None) -> dict:
    """Describe a single-run delta without making a reproducibility claim."""
    before = baseline.get("value") if baseline else None
    after = candidate.get("value") if candidate else None
    delta = after - before if before is not None and after is not None else None
    percent = delta / before * 100 if delta is not None and before else None
    threshold = practical_threshold_pct(metric)
    dispersions = [side.get("dispersion") if side else None for side in (baseline, candidate)]
    counts = [side.get("sample_count") if side else None for side in (baseline, candidate)]
    uncertainty = "recorded" if (
        all(value is not None for value in dispersions)
        and all(isinstance(count, int) and count >= 2 for count in counts)
    ) else "insufficient"
    clears = abs(percent) >= threshold if percent is not None else None
    return {
        "baseline": before,
        "candidate": after,
        "delta": delta,
        "percent_change": percent,
        "practical_threshold_pct": threshold,
        "clears_practical_threshold": clears,
        "within_run_uncertainty": uncertainty,
        "baseline_dispersion": dispersions[0],
        "candidate_dispersion": dispersions[1],
        "baseline_samples": counts[0],
        "candidate_samples": counts[1],
        "verdict": "repeated_trials_required",
    }
