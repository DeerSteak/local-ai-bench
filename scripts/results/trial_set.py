"""Compatible independent-trial aggregation and regression verdicts."""

import math
import statistics

from scripts.results.canonical_json import sha256_json
from scripts.results.result_history import compare_results, extract_comparable_metrics, hardware_identity
from scripts.results.significance import practical_threshold_pct


TRIAL_SET_SCHEMA_VERSION = 1
MIN_INTERVAL_TRIALS = 5
T_CRITICAL_95 = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
                 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
                 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
                 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
                 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
                 30: 2.042}


def _t_critical(degrees_freedom: int) -> float:
    return T_CRITICAL_95.get(degrees_freedom, 1.96 if degrees_freedom > 30 else math.inf)


def monotonic_drift(values: list[float]) -> str:
    """Flag sustained ordinal movement without relabeling flat series as drift."""
    if len(values) < 3:
        return "insufficient"
    declining = all(left >= right for left, right in zip(values, values[1:]))
    increasing = all(left <= right for left, right in zip(values, values[1:]))
    if declining and values[0] > values[-1]:
        return "declining"
    if increasing and values[0] < values[-1]:
        return "increasing"
    return "none"


def aggregate_trials(values: list[float]) -> dict:
    """Describe independent trials and add a t interval only at the declared minimum."""
    count = len(values)
    result = {
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "stdev": statistics.stdev(values) if count >= 2 else None,
        "trial_count": count,
        "interval": None,
        "interval_method": "student_t_95" if count >= MIN_INTERVAL_TRIALS else None,
        "drift": monotonic_drift(values),
    }
    if count >= MIN_INTERVAL_TRIALS:
        margin = _t_critical(count - 1) * result["stdev"] / math.sqrt(count)
        result["interval"] = [result["mean"] - margin, result["mean"] + margin]
    return result


def trial_set_compatibility(results: list[dict]) -> dict:
    """Reuse single-run compatibility and hardware identity for pooling."""
    if not results:
        return {"compatible": False, "incompatible_fields": ["empty_trial_set"]}
    first = results[0]
    first_hardware = hardware_identity(first.get("profile") or {})
    incompatible = set()
    for candidate in results[1:]:
        comparison = compare_results(first, candidate)
        incompatible.update(comparison["incompatible_fields"])
        if hardware_identity(candidate.get("profile") or {}) != first_hardware:
            incompatible.add("hardware_identity")
    return {"compatible": not incompatible, "incompatible_fields": sorted(incompatible)}


def _relative_change(before: float, after: float) -> float | None:
    return (after - before) / before * 100 if before else None


def _verdict(interval: list[float] | None, threshold: float, higher_is_better: bool,
             drift: bool) -> str:
    if interval is None or drift:
        return "inconclusive"
    low, high = interval
    if low >= threshold:
        return "improved" if higher_is_better else "regressed"
    if high <= -threshold:
        return "regressed" if higher_is_better else "improved"
    if low >= -threshold and high <= threshold:
        return "unchanged"
    return "inconclusive"


def _independent_change_interval(baseline: list[float], candidate: list[float]) -> list[float] | None:
    if len(baseline) < MIN_INTERVAL_TRIALS or len(candidate) < MIN_INTERVAL_TRIALS:
        return None
    baseline_mean = statistics.mean(baseline)
    if baseline_mean == 0:
        return None
    baseline_term = statistics.variance(baseline) / len(baseline)
    candidate_term = statistics.variance(candidate) / len(candidate)
    standard_error = math.sqrt(baseline_term + candidate_term)
    if standard_error == 0:
        change = _relative_change(baseline_mean, statistics.mean(candidate))
        return [change, change] if change is not None else None
    numerator = (baseline_term + candidate_term) ** 2
    denominator = (baseline_term ** 2 / (len(baseline) - 1)
                   + candidate_term ** 2 / (len(candidate) - 1))
    degrees_freedom = max(1, round(numerator / denominator))
    difference = statistics.mean(candidate) - baseline_mean
    margin = _t_critical(degrees_freedom) * standard_error
    return [(difference - margin) / baseline_mean * 100,
            (difference + margin) / baseline_mean * 100]


def analyze_trial_metric(metric: str, baseline: list[float], candidate: list[float],
                         *, paired: bool) -> dict:
    """Compare one metric across compatible independent trial groups."""
    baseline_stats = aggregate_trials(baseline)
    candidate_stats = aggregate_trials(candidate)
    changes = []
    mode = "paired" if paired and len(baseline) == len(candidate) else "independent"
    if mode == "paired":
        changes = [change for before, after in zip(baseline, candidate)
                   if (change := _relative_change(before, after)) is not None]
    elif baseline_stats["mean"] is not None and candidate_stats["mean"] is not None:
        change = _relative_change(baseline_stats["mean"], candidate_stats["mean"])
        if change is not None:
            changes = [change]
    change_stats = aggregate_trials(changes)
    change_interval = change_stats["interval"] if mode == "paired" \
        else _independent_change_interval(baseline, candidate)
    interval_method = change_stats["interval_method"] if mode == "paired" \
        else "welch_t_95" if change_interval is not None else None
    threshold = practical_threshold_pct(metric)
    drift = baseline_stats["drift"] not in {"none", "insufficient"} \
        or candidate_stats["drift"] not in {"none", "insufficient"}
    higher_is_better = metric not in {
        "ttft_mean_sec", "client_ttft_mean_sec", "server_prompt_mean_sec",
        "sec_per_image_mean",
    }
    return {
        "metric": metric,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "relative_changes_pct": changes,
        "change_interval_pct": change_interval,
        "interval_method": interval_method,
        "comparison_mode": mode,
        "practical_threshold_pct": threshold,
        "verdict": _verdict(change_interval, threshold, higher_is_better, drift),
    }


def build_trial_set(baseline: list[dict], candidate: list[dict]) -> dict:
    """Build a versioned comparison artifact from two compatible trial groups."""
    baseline_compatibility = trial_set_compatibility(baseline)
    candidate_compatibility = trial_set_compatibility(candidate)
    cross = trial_set_compatibility([*baseline, *candidate])
    if not baseline_compatibility["compatible"] or not candidate_compatibility["compatible"] \
            or not cross["compatible"]:
        fields = set(baseline_compatibility["incompatible_fields"])
        fields.update(candidate_compatibility["incompatible_fields"])
        fields.update(cross["incompatible_fields"])
        raise ValueError(f"incompatible trial set: {', '.join(sorted(fields))}")
    baseline_metrics = [extract_comparable_metrics(result) for result in baseline]
    candidate_metrics = [extract_comparable_metrics(result) for result in candidate]
    baseline_keys = [tuple(metrics) for metrics in baseline_metrics]
    candidate_keys = [tuple(metrics) for metrics in candidate_metrics]
    paired = len(baseline) == len(candidate) and baseline_keys == candidate_keys
    keys = sorted(set.intersection(*(set(metrics) for metrics in [*baseline_metrics, *candidate_metrics])))
    rows = []
    for key in keys:
        metric = key.rsplit("/", 1)[-1]
        rows.append({"key": key, **analyze_trial_metric(
            metric,
            [metrics[key]["value"] for metrics in baseline_metrics],
            [metrics[key]["value"] for metrics in candidate_metrics],
            paired=paired,
        )})
    return {
        "schema_version": TRIAL_SET_SCHEMA_VERSION,
        "compatible": True,
        "comparison_mode": "paired" if paired else "independent",
        "baseline_trials": len(baseline),
        "candidate_trials": len(candidate),
        "methodology_profile": baseline[0]["run"]["plan"]["effective_config"].get(
            "methodology_profile"),
        "hardware_identity": hardware_identity(baseline[0].get("profile") or {}),
        "source_sha256": {
            "baseline": [sha256_json(result) for result in baseline],
            "candidate": [sha256_json(result) for result in candidate],
        },
        "rows": rows,
    }
