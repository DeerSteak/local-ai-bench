"""Constraint-first recommendation evaluation over compatible benchmark evidence."""

from dataclasses import asdict, dataclass
import math

from scripts.results.canonical_json import sha256_json
from scripts.results.result_store import as_dict
from scripts.results.trial_set import MIN_INTERVAL_TRIALS, analyze_trial_metric, trial_set_compatibility


RECOMMENDATION_SCHEMA_VERSION = 1
SUPPORTED_WORKLOADS = {
    "llm", "llm_conversation", "concurrency_tool", "concurrency_chat", "images",
}
SUPPORTED_ACCURACY_SECTIONS = {"mcq", "math", "reasoning", "code", "tool"}
SUPPORTED_OBJECTIVES = {"accuracy", "throughput", "ttft", "memory", "efficiency"}
CONSTRAINT_FIELDS = {
    "workload", "case", "accuracy_section", "primary_objective", "minimum_accuracy_pct",
    "maximum_ttft_sec", "minimum_throughput", "concurrency", "maximum_memory_gb",
    "minimum_memory_headroom_gb", "minimum_efficiency_per_joule",
}


@dataclass(frozen=True)
class ConstraintSet:
    workload: str
    case: str | None = None
    accuracy_section: str | None = None
    primary_objective: str = "throughput"
    minimum_accuracy_pct: float | None = None
    maximum_ttft_sec: float | None = None
    minimum_throughput: float | None = None
    concurrency: int | None = None
    maximum_memory_gb: float | None = None
    minimum_memory_headroom_gb: float | None = None
    minimum_efficiency_per_joule: float | None = None


def _finite(value: object, field: str, *, minimum: float = 0.0) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{field} must be a finite number at least {minimum:g}")
    return number


def parse_constraints(value: dict) -> ConstraintSet:
    """Validate the explicit request without turning absent constraints into zero."""
    if not isinstance(value, dict):
        raise ValueError("recommendation constraints must be an object")
    unknown = sorted(set(value) - CONSTRAINT_FIELDS)
    if unknown:
        raise ValueError(f"unknown recommendation constraint fields: {', '.join(unknown)}")
    workload = value.get("workload")
    if workload not in SUPPORTED_WORKLOADS:
        raise ValueError(f"workload must be one of: {', '.join(sorted(SUPPORTED_WORKLOADS))}")
    objective = value.get("primary_objective", "throughput")
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"primary_objective must be one of: {', '.join(sorted(SUPPORTED_OBJECTIVES))}")
    accuracy_section = value.get("accuracy_section")
    if accuracy_section is not None and accuracy_section not in SUPPORTED_ACCURACY_SECTIONS:
        raise ValueError(
            f"accuracy_section must be one of: {', '.join(sorted(SUPPORTED_ACCURACY_SECTIONS))}")
    case = value.get("case")
    if case is not None and (not isinstance(case, str) or not case.strip()):
        raise ValueError("case must be a non-empty string")
    concurrency = value.get("concurrency")
    if concurrency is not None and (isinstance(concurrency, bool)
                                    or not isinstance(concurrency, int) or concurrency < 1):
        raise ValueError("concurrency must be an integer at least 1")
    minimum_accuracy = _finite(value.get("minimum_accuracy_pct"), "minimum_accuracy_pct")
    if minimum_accuracy is not None and minimum_accuracy > 100:
        raise ValueError("minimum_accuracy_pct must be at most 100")
    constraints = ConstraintSet(
        workload=workload,
        case=case,
        accuracy_section=accuracy_section,
        primary_objective=objective,
        minimum_accuracy_pct=minimum_accuracy,
        maximum_ttft_sec=_finite(value.get("maximum_ttft_sec"), "maximum_ttft_sec"),
        minimum_throughput=_finite(value.get("minimum_throughput"), "minimum_throughput"),
        concurrency=concurrency,
        maximum_memory_gb=_finite(value.get("maximum_memory_gb"), "maximum_memory_gb"),
        minimum_memory_headroom_gb=_finite(
            value.get("minimum_memory_headroom_gb"), "minimum_memory_headroom_gb"),
        minimum_efficiency_per_joule=_finite(
            value.get("minimum_efficiency_per_joule"), "minimum_efficiency_per_joule"),
    )
    if constraints.minimum_accuracy_pct is not None and accuracy_section is None:
        raise ValueError("accuracy_section is required when minimum_accuracy_pct is set")
    if objective == "accuracy" and accuracy_section is None:
        raise ValueError("accuracy_section is required for the accuracy objective")
    if concurrency is not None and workload not in {"concurrency_tool", "concurrency_chat"}:
        raise ValueError("concurrency is only valid for a concurrency workload")
    return constraints


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _case_values(result: dict, constraints: ConstraintSet, model: str) -> tuple[dict, str]:
    section = as_dict(result.get(constraints.workload))
    model_values = as_dict(section.get(model))
    case = str(constraints.concurrency) if constraints.concurrency is not None else constraints.case
    if constraints.workload == "images":
        model_values = as_dict(model_values.get("resolutions"))
    if case is None:
        return {}, f"{constraints.workload}/{model}/<case>"
    return as_dict(model_values.get(case)), f"{constraints.workload}/{model}/{case}"


def _measurement(value: float | None, path: str, unit: str,
                 raw_evidence_paths: list[str]) -> dict | None:
    return {
        "value": value, "unit": unit, "evidence_path": path,
        "raw_evidence_paths": raw_evidence_paths,
    } if value is not None else None


def candidate_evidence(result: dict, constraints: ConstraintSet, model: str) -> dict:
    """Extract only named measurements needed by recommendation policy."""
    values, case_path = _case_values(result, constraints, model)
    throughput_key = "aggregate_tps" if constraints.workload.startswith("concurrency_") \
        else "tps_mean"
    if constraints.workload == "images":
        throughput_key = "sec_per_image_mean"
    ttft_key = "client_ttft_mean_sec" if constraints.workload == "llm_conversation" \
        else "ttft_mean_sec"
    memory = as_dict(values.get("memory"))
    memory_summary = as_dict(memory.get("summary"))
    process_memory = as_dict(memory_summary.get("process_rss_gb"))
    accelerator_memory = as_dict(memory_summary.get("accelerator_memory_used_gb"))
    peak_memory = _number(accelerator_memory.get("peak_gb"))
    memory_channel = "accelerator_memory_used_gb"
    if peak_memory is None:
        peak_memory = _number(process_memory.get("peak_gb"))
        memory_channel = "process_rss_gb"
    headroom = as_dict(memory.get("headroom"))
    power = as_dict(values.get("power"))
    efficiency = as_dict(power.get("efficiency"))
    accuracy_values = as_dict(as_dict(result.get(constraints.accuracy_section)).get(model)) \
        if constraints.accuracy_section else {}
    return {
        "throughput": _measurement(
            _number(values.get(throughput_key)), f"{case_path}/{throughput_key}",
            "seconds_per_image" if constraints.workload == "images" else "tokens_per_second",
            [f"{case_path}/valid_samples"]),
        "ttft": _measurement(
            _number(values.get(ttft_key)), f"{case_path}/{ttft_key}", "seconds",
            [f"{case_path}/valid_samples"]),
        "accuracy": _measurement(
            _number(accuracy_values.get("accuracy_pct")),
            f"{constraints.accuracy_section}/{model}/accuracy_pct", "percent",
            [f"answers_{constraints.accuracy_section}/{model}"]),
        "memory": _measurement(
            peak_memory, f"{case_path}/memory/summary/{memory_channel}/peak_gb", "GB",
            [f"{case_path}/memory/windows"]),
        "memory_headroom": _measurement(
            _number(headroom.get("absolute_gb")),
            f"{case_path}/memory/headroom/absolute_gb", "GB",
            [f"{case_path}/memory/windows"]),
        "efficiency": _measurement(
            _number(efficiency.get("per_joule")),
            f"{case_path}/power/efficiency/per_joule", str(efficiency.get("unit") or "per_joule"),
            [f"{case_path}/power/windows"]),
    }


def _candidate_models(result: dict, constraints: ConstraintSet) -> list[str]:
    models = set(as_dict(result.get(constraints.workload)))
    if constraints.accuracy_section:
        models.update(as_dict(result.get(constraints.accuracy_section)))
    return sorted(str(model) for model in models)


def _requirements(constraints: ConstraintSet) -> list[tuple[str, str, float, str]]:
    checks = []
    for metric, operator, threshold in (
        ("accuracy", "minimum", constraints.minimum_accuracy_pct),
        ("ttft", "maximum", constraints.maximum_ttft_sec),
        ("throughput", "minimum", constraints.minimum_throughput),
        ("memory", "maximum", constraints.maximum_memory_gb),
        ("memory_headroom", "minimum", constraints.minimum_memory_headroom_gb),
        ("efficiency", "minimum", constraints.minimum_efficiency_per_joule),
    ):
        if threshold is not None:
            checks.append((metric, operator, threshold, metric))
    if not any(metric == constraints.primary_objective for metric, *_ in checks):
        checks.append((constraints.primary_objective, "ranking", 0.0, constraints.primary_objective))
    return checks


def _mean_measurement(measurements: list[dict]) -> dict:
    first = measurements[0]
    return {
        "value": sum(item["value"] for item in measurements) / len(measurements),
        "unit": first["unit"],
        "evidence_path": first["evidence_path"],
        "raw_evidence_paths": first["raw_evidence_paths"],
        "trial_values": [item["value"] for item in measurements],
    }


def _trial_metric(constraints: ConstraintSet) -> str | None:
    if constraints.primary_objective == "accuracy":
        return "accuracy_pct"
    if constraints.primary_objective == "ttft":
        return "client_ttft_mean_sec" if constraints.workload == "llm_conversation" \
            else "ttft_mean_sec"
    if constraints.primary_objective == "throughput":
        if constraints.workload == "images":
            return "sec_per_image_mean"
        return "aggregate_tps" if constraints.workload.startswith("concurrency_") else "tps_mean"
    return None


def _rank_with_trials(eligible: list[dict], constraints: ConstraintSet) -> tuple[str, list[dict]]:
    metric = _trial_metric(constraints)
    if metric is None or any(len(item["evidence"][constraints.primary_objective]["trial_values"])
                             < MIN_INTERVAL_TRIALS for item in eligible):
        return "insufficient_evidence", []
    best = eligible[0]
    comparisons = []
    for candidate in eligible[1:]:
        comparisons.append(analyze_trial_metric(
            metric,
            candidate["evidence"][constraints.primary_objective]["trial_values"],
            best["evidence"][constraints.primary_objective]["trial_values"],
            paired=True,
        ))
    verdicts = [comparison["verdict"] for comparison in comparisons]
    if all(verdict in {"unchanged", "improved"} for verdict in verdicts):
        tied = [best, *(candidate for candidate, comparison in zip(eligible[1:], comparisons)
                        if comparison["verdict"] == "unchanged")]
        if len(tied) > 1:
            return "tied", tied
        return "recommended", [best]
    return "insufficient_evidence", []


def _result_evidence_gaps(result: dict) -> list[str]:
    run = as_dict(result.get("run"))
    plan = as_dict(run.get("plan"))
    settings = as_dict(plan.get("effective_config"))
    gaps = []
    if run.get("status") != "complete":
        gaps.append("complete_run")
    if not settings.get("methodology_profile"):
        gaps.append("methodology_profile")
    return gaps


def evaluate_recommendation(result: dict | list[dict], request: dict) -> dict:
    """Filter candidates before ranking and preserve every exclusion or evidence gap."""
    constraints = parse_constraints(request)
    results = result if isinstance(result, list) else [result]
    if not results or any(not isinstance(item, dict) for item in results):
        raise ValueError("recommendation requires at least one result object")
    compatibility = trial_set_compatibility(results)
    if not compatibility["compatible"]:
        raise ValueError(
            f"incompatible recommendation evidence: {', '.join(compatibility['incompatible_fields'])}")
    eligible, eliminated, unevaluated = [], [], []
    models = sorted(set().union(*(_candidate_models(item, constraints) for item in results)))
    result_gaps = sorted(set().union(*(_result_evidence_gaps(item) for item in results)))
    for model in models:
        trial_evidence = [candidate_evidence(item, constraints, model) for item in results]
        evidence = {}
        for metric in trial_evidence[0]:
            measurements = [item[metric] for item in trial_evidence]
            evidence[metric] = _mean_measurement(measurements) \
                if all(measurement is not None for measurement in measurements) else None
        missing = sorted({metric for metric, *_ in _requirements(constraints)
                          if evidence.get(metric) is None} | set(result_gaps))
        if missing:
            unevaluated.append({
                "candidate": model,
                "missing_evidence": missing,
                "resolution": {
                    "workload": constraints.workload,
                    "case": str(constraints.concurrency) if constraints.concurrency else constraints.case,
                    "accuracy_section": constraints.accuracy_section,
                },
            })
            continue
        failures = []
        for metric, operator, threshold, _ in _requirements(constraints):
            if operator == "ranking":
                continue
            measurement = evidence[metric]
            failed = measurement["value"] < threshold if operator == "minimum" \
                else measurement["value"] > threshold
            if failed:
                failures.append({
                    "constraint": metric, "operator": operator, "threshold": threshold,
                    "measurement": measurement,
                })
        if failures:
            eliminated.append({"candidate": model, "reasons": failures})
        else:
            eligible.append({"candidate": model, "evidence": evidence})
    objective = constraints.primary_objective
    reverse = objective not in {"ttft", "memory"} and constraints.workload != "images"
    eligible.sort(key=lambda item: item["evidence"][objective]["value"], reverse=reverse)
    if not eligible:
        verdict = "insufficient_evidence"
        ranked = []
    elif len(eligible) == 1:
        verdict = "recommended"
        ranked = eligible
    else:
        verdict, ranked = _rank_with_trials(eligible, constraints)
        if verdict == "insufficient_evidence":
            for item in eligible:
                unevaluated.append({
                    "candidate": item["candidate"],
                    "missing_evidence": ["qualified_repeated_trial_verdict"],
                    "resolution": {
                        "objective": objective,
                        "evidence_path": item["evidence"][objective]["evidence_path"],
                        "minimum_compatible_trials": MIN_INTERVAL_TRIALS,
                    },
                })
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "artifact_type": "recommendation",
        "constraints": asdict(constraints),
        "source_sha256": [sha256_json(item) for item in results],
        "verdict": verdict,
        "recommended": ranked[:1] if verdict == "recommended" else [],
        "tied": ranked if verdict == "tied" else [],
        "eliminated": eliminated,
        "unevaluated": unevaluated,
    }


def validate_recommendation_artifact(artifact: dict, *, source_result: dict | None = None) -> None:
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "recommendation" \
            or artifact.get("schema_version") != RECOMMENDATION_SCHEMA_VERSION:
        raise ValueError("unsupported recommendation artifact")
    if artifact.get("verdict") not in {"recommended", "tied", "insufficient_evidence"}:
        raise ValueError("invalid recommendation verdict")
    parse_constraints(as_dict(artifact.get("constraints")))
    for field in ("source_sha256", "recommended", "tied", "eliminated", "unevaluated"):
        if not isinstance(artifact.get(field), list):
            raise ValueError(f"recommendation artifact field {field} must be a list")
    if source_result is not None and sha256_json(source_result) not in artifact["source_sha256"]:
        raise ValueError("recommendation artifact does not cite this result")
