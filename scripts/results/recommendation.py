"""Constraint-first recommendation evaluation over compatible benchmark evidence."""

from dataclasses import dataclass
import math

from scripts.results.canonical_json import sha256_json
from scripts.results.result_store import as_dict


RECOMMENDATION_SCHEMA_VERSION = 1
SUPPORTED_WORKLOADS = {
    "llm", "llm_conversation", "concurrency_tool", "concurrency_chat", "images",
}
SUPPORTED_ACCURACY_SECTIONS = {"mcq", "math", "reasoning", "code", "tool"}
SUPPORTED_OBJECTIVES = {"accuracy", "throughput", "ttft", "memory", "efficiency"}


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


def _measurement(value: float | None, path: str, unit: str) -> dict | None:
    return {"value": value, "unit": unit, "evidence_path": path} if value is not None else None


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
            "seconds_per_image" if constraints.workload == "images" else "tokens_per_second"),
        "ttft": _measurement(
            _number(values.get(ttft_key)), f"{case_path}/{ttft_key}", "seconds"),
        "accuracy": _measurement(
            _number(accuracy_values.get("accuracy_pct")),
            f"{constraints.accuracy_section}/{model}/accuracy_pct", "percent"),
        "memory": _measurement(
            peak_memory, f"{case_path}/memory/summary/{memory_channel}/peak_gb", "GB"),
        "memory_headroom": _measurement(
            _number(headroom.get("absolute_gb")),
            f"{case_path}/memory/headroom/absolute_gb", "GB"),
        "efficiency": _measurement(
            _number(efficiency.get("per_joule")),
            f"{case_path}/power/efficiency/per_joule", str(efficiency.get("unit") or "per_joule")),
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


def evaluate_recommendation(result: dict, request: dict) -> dict:
    """Filter candidates before ranking and preserve every exclusion or evidence gap."""
    constraints = parse_constraints(request)
    eligible, eliminated, unevaluated = [], [], []
    for model in _candidate_models(result, constraints):
        evidence = candidate_evidence(result, constraints, model)
        missing = sorted({metric for metric, *_ in _requirements(constraints)
                          if evidence.get(metric) is None})
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
        # Independent means cannot establish a close ordering; trial evidence is added separately.
        verdict = "insufficient_evidence"
        ranked = []
        for item in eligible:
            unevaluated.append({
                "candidate": item["candidate"],
                "missing_evidence": ["qualified_repeated_trial_verdict"],
                "resolution": {"objective": objective, "evidence_path": item["evidence"][objective]["evidence_path"]},
            })
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "artifact_type": "recommendation",
        "constraints": request,
        "source_sha256": [sha256_json(result)],
        "verdict": verdict,
        "recommended": ranked[:1] if verdict == "recommended" else [],
        "tied": [],
        "eliminated": eliminated,
        "unevaluated": unevaluated,
    }
