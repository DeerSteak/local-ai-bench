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
    if concurrency is not None and case is not None:
        raise ValueError("set concurrency or case, not both")
    if concurrency is None and case is None:
        raise ValueError("case is required unless concurrency is set")
    if workload == "images" and accuracy_section is not None:
        raise ValueError("accuracy_section is not available for image recommendations")
    return constraints


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _case_values(result: dict, constraints: ConstraintSet, model: str) -> tuple[dict, dict, str]:
    section = as_dict(result.get(constraints.workload))
    model_values = as_dict(section.get(model))
    case = str(constraints.concurrency) if constraints.concurrency is not None else constraints.case
    if constraints.workload == "images":
        resolutions = as_dict(model_values.get("resolutions"))
        return (as_dict(resolutions.get(case)), model_values,
                f"images/{model}/resolutions/{case}")
    return as_dict(model_values.get(case)), model_values, \
        f"{constraints.workload}/{model}/{case}"


def _measurement(value: float | None, path: str, unit: str,
                 raw_evidence_paths: list[str], **provenance) -> dict | None:
    measurement = {
        "value": value, "unit": unit, "evidence_path": path,
        "raw_evidence_paths": raw_evidence_paths,
    }
    measurement.update(provenance)
    return measurement if value is not None else None


def _raw_evidence_path(values: dict, key: str, path: str) -> list[str]:
    return [path] if isinstance(values.get(key), list) else []


def candidate_evidence(result: dict, constraints: ConstraintSet, model: str) -> dict:
    """Extract only named measurements needed by recommendation policy."""
    values, model_values, case_path = _case_values(result, constraints, model)
    throughput_key = "aggregate_tps" if constraints.workload.startswith("concurrency_") \
        else "tps_mean"
    if constraints.workload == "images":
        throughput_key = "sec_per_image_mean"
    ttft_key = "client_ttft_mean_sec" if constraints.workload == "llm_conversation" \
        else "ttft_mean_sec"
    telemetry_values = model_values if constraints.workload == "images" else values
    telemetry_path = f"{constraints.workload}/{model}" \
        if constraints.workload == "images" else case_path
    memory = as_dict(telemetry_values.get("memory"))
    memory_summary = as_dict(memory.get("summary"))
    process_memory = as_dict(memory_summary.get("process_rss_gb"))
    accelerator_memory = as_dict(memory_summary.get("accelerator_memory_used_gb"))
    peak_memory = _number(accelerator_memory.get("peak_gb"))
    memory_channel = "accelerator_memory_used_gb"
    if peak_memory is None:
        peak_memory = _number(process_memory.get("peak_gb"))
        memory_channel = "process_rss_gb"
    headroom = as_dict(memory.get("headroom"))
    power = as_dict(telemetry_values.get("power"))
    efficiency = as_dict(power.get("efficiency"))
    accuracy_values = as_dict(as_dict(result.get(constraints.accuracy_section)).get(model)) \
        if constraints.accuracy_section else {}
    throughput = _number(values.get(throughput_key))
    throughput_provenance = {}
    if constraints.workload == "images" and throughput:
        throughput_provenance = {
            "source_value": throughput,
            "source_unit": "seconds_per_image",
            "derivation": "reciprocal",
        }
        throughput = round(1 / throughput, 4)
    raw_samples_key = "runs" if constraints.workload == "images" else "valid_samples"
    return {
        "throughput": _measurement(
            throughput, f"{case_path}/{throughput_key}",
            "images_per_second" if constraints.workload == "images" else "tokens_per_second",
            _raw_evidence_path(values, raw_samples_key, f"{case_path}/{raw_samples_key}"),
            **throughput_provenance),
        "ttft": _measurement(
            _number(values.get(ttft_key)), f"{case_path}/{ttft_key}", "seconds",
            _raw_evidence_path(values, "valid_samples", f"{case_path}/valid_samples")),
        "accuracy": _measurement(
            _number(accuracy_values.get("accuracy_pct")),
            f"{constraints.accuracy_section}/{model}/accuracy_pct", "percent",
            [f"answers_{constraints.accuracy_section}/{model}"]),
        "memory": _measurement(
            peak_memory, f"{telemetry_path}/memory/summary/{memory_channel}/peak_gb", "GB",
            _raw_evidence_path(memory, "windows", f"{telemetry_path}/memory/windows")),
        "memory_headroom": _measurement(
            _number(headroom.get("absolute_gb")),
            f"{telemetry_path}/memory/headroom/absolute_gb", "GB",
            _raw_evidence_path(memory, "windows", f"{telemetry_path}/memory/windows")),
        "efficiency": _measurement(
            _number(efficiency.get("per_joule")),
            f"{telemetry_path}/power/efficiency/per_joule", str(efficiency.get("unit") or "per_joule"),
            _raw_evidence_path(power, "windows", f"{telemetry_path}/power/windows")),
    }


def _candidate_models(result: dict, constraints: ConstraintSet) -> list[str]:
    models = set(as_dict(result.get(constraints.workload)))
    if constraints.accuracy_section:
        models.update(as_dict(result.get(constraints.accuracy_section)))
    return sorted(str(model) for model in models)


def _available_cases(results: list[dict], constraints: ConstraintSet) -> list[str]:
    cases = set()
    for result in results:
        for model_values in as_dict(result.get(constraints.workload)).values():
            values = as_dict(model_values)
            if constraints.workload == "images":
                values = as_dict(values.get("resolutions"))
            cases.update(str(case) for case, case_values in values.items()
                         if isinstance(case_values, dict))
    return sorted(cases)


def _validate_requested_case(results: list[dict], constraints: ConstraintSet) -> None:
    requested = str(constraints.concurrency) if constraints.concurrency is not None \
        else constraints.case
    available = _available_cases(results, constraints)
    if available and requested not in available:
        raise ValueError(
            f"unknown case {requested!r} for {constraints.workload}; "
            f"available cases: {', '.join(available)}")


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
    mean = sum(item["value"] for item in measurements) / len(measurements)
    result = {
        "value": round(mean, 4) if first["unit"] == "images_per_second" else mean,
        "unit": first["unit"],
        "evidence_path": first["evidence_path"],
        "raw_evidence_paths": first["raw_evidence_paths"],
        "trial_values": [item["value"] for item in measurements],
    }
    if "source_value" in first:
        source_values = [item["source_value"] for item in measurements]
        result.update({
            "source_value": round(sum(source_values) / len(source_values), 4),
            "source_trial_values": source_values,
            "source_unit": first["source_unit"],
            "derivation": first["derivation"],
        })
    return result


def _trial_metric(constraints: ConstraintSet) -> str | None:
    if constraints.primary_objective == "accuracy":
        return "accuracy_pct"
    if constraints.primary_objective == "ttft":
        return "client_ttft_mean_sec" if constraints.workload == "llm_conversation" \
            else "ttft_mean_sec"
    if constraints.primary_objective == "throughput":
        if constraints.workload == "images":
            return None
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
    digests = [sha256_json(item) for item in results]
    if len(digests) != len(set(digests)):
        raise ValueError("recommendation trials must contain distinct independent result files")
    compatibility = trial_set_compatibility(results)
    if not compatibility["compatible"]:
        raise ValueError(
            f"incompatible recommendation evidence: {', '.join(compatibility['incompatible_fields'])}")
    _validate_requested_case(results, constraints)
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
    reverse = objective not in {"ttft", "memory"}
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
    ranked_candidates = {item["candidate"] for item in ranked}
    other_eligible = [item for item in eligible if item["candidate"] not in ranked_candidates] \
        if verdict != "insufficient_evidence" else []
    if verdict != "insufficient_evidence":
        resolution = None
    elif unevaluated:
        resolution = {"action": "run_missing_evidence", "candidate_gaps": len(unevaluated)}
    elif eliminated:
        resolution = {"action": "test_other_candidates_or_change_constraints"}
    else:
        resolution = {
            "action": "run_missing_evidence", "workload": constraints.workload,
            "case": str(constraints.concurrency) if constraints.concurrency else constraints.case,
        }
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "artifact_type": "recommendation",
        "constraints": asdict(constraints),
        "source_sha256": digests,
        "verdict": verdict,
        "candidates": {
            "recommended": ranked[:1] if verdict == "recommended" else [],
            "tied": ranked if verdict == "tied" else [],
            "other_eligible": other_eligible,
            "eliminated": eliminated,
            "unevaluated": unevaluated,
        },
        "resolution": resolution,
    }


def validate_recommendation_artifact(artifact: dict, *, source_result: dict | None = None) -> None:
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "recommendation" \
            or artifact.get("schema_version") != RECOMMENDATION_SCHEMA_VERSION:
        raise ValueError("unsupported recommendation artifact")
    if artifact.get("verdict") not in {"recommended", "tied", "insufficient_evidence"}:
        raise ValueError("invalid recommendation verdict")
    parse_constraints(as_dict(artifact.get("constraints")))
    if not isinstance(artifact.get("source_sha256"), list):
        raise ValueError("recommendation artifact field source_sha256 must be a list")
    candidates = as_dict(artifact.get("candidates"))
    for field in ("recommended", "tied", "other_eligible", "eliminated", "unevaluated"):
        if not isinstance(candidates.get(field), list):
            raise ValueError(f"recommendation candidate group {field} must be a list")
    expected = {
        "recommended": (len(candidates["recommended"]) == 1 and not candidates["tied"]),
        "tied": (len(candidates["tied"]) >= 2 and not candidates["recommended"]),
        "insufficient_evidence": (not candidates["recommended"] and not candidates["tied"]
                                  and not candidates["other_eligible"]),
    }
    if not expected[artifact["verdict"]]:
        raise ValueError("recommendation verdict does not match its candidate groups")
    if artifact["verdict"] == "insufficient_evidence" \
            and not isinstance(artifact.get("resolution"), dict):
        raise ValueError("insufficient recommendation must include a resolution")
    candidate_names = []
    for field in ("recommended", "tied", "other_eligible", "eliminated", "unevaluated"):
        candidate_names.extend(
            item.get("candidate") for item in candidates[field] if isinstance(item, dict))
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("a recommendation candidate must appear in exactly one outcome group")
    if source_result is not None and sha256_json(source_result) not in artifact["source_sha256"]:
        raise ValueError("recommendation artifact does not cite this result")
