"""Pure cross-quantization tradeoff analysis over one compatible result."""

import math

from scripts.results.result_store import as_dict


VARIANT_COMPARISON_SCHEMA = 1
QUALITY_VERDICTS = {"improved", "regressed", "unchanged", "inconclusive"}


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value) else None


def _peak_memory(case: dict) -> float | None:
    summary = as_dict(as_dict(case.get("memory")).get("summary"))
    accelerator = as_dict(summary.get("accelerator_memory_used_gb"))
    process = as_dict(summary.get("process_rss_gb"))
    accelerator_peak = _number(accelerator.get("peak_gb"))
    return accelerator_peak if accelerator_peak is not None else _number(process.get("peak_gb"))


def _metric(value: float | None, reference: float | None, *, percentage_points=False) -> dict:
    if value is None or reference is None:
        return {"value": value, "delta": None, "status": "unavailable"}
    if percentage_points:
        delta = value - reference
        unit = "percentage_points"
    elif reference == 0:
        return {"value": value, "delta": None, "status": "indeterminate"}
    else:
        delta = (value - reference) / reference * 100
        unit = "percent"
    return {"value": value, "delta": round(delta, 4), "delta_unit": unit, "status": "measured"}


def build_variant_comparison(result: dict, *, base_model: str, reference_variant: str,
                             performance_section: str, case: str,
                             accuracy_section: str, quality_verdicts: dict[str, str] | None = None) -> dict:
    identities = [
        model for model in as_dict(as_dict(as_dict(result.get("run")).get("plan")).get("models"))
        .get("llm", []) if isinstance(model, dict) and model.get("base_model") == base_model
    ]
    if not identities:
        raise ValueError(f"result contains no variants for base model: {base_model}")
    variants = [model.get("variant") for model in identities]
    if any(not isinstance(variant, str) or not variant for variant in variants) \
            or len(variants) != len(set(variants)):
        raise ValueError("variant comparison requires distinct explicit variant identities")
    if reference_variant not in variants:
        raise ValueError(f"reference variant is not present: {reference_variant}")
    verdicts = quality_verdicts or {}
    unknown_verdicts = set(verdicts.values()) - QUALITY_VERDICTS
    if unknown_verdicts:
        raise ValueError(f"unknown quality verdict: {sorted(unknown_verdicts)[0]}")

    performance = as_dict(result.get(performance_section))
    accuracy = as_dict(result.get(accuracy_section))

    def measurements(identity: dict) -> dict[str, float | None]:
        short = identity["short"]
        case_values = as_dict(as_dict(performance.get(short)).get(case))
        return {
            "quality": _number(as_dict(accuracy.get(short)).get("accuracy_pct")),
            "throughput": _number(case_values.get("tps_mean")),
            "memory": _peak_memory(case_values),
            "energy": _number(as_dict(case_values.get("power")).get("energy_joules")),
        }

    measured = {identity["variant"]: measurements(identity) for identity in identities}
    reference = measured[reference_variant]
    rows = []
    for identity in identities:
        variant = identity["variant"]
        values = measured[variant]
        verdict = "reference" if variant == reference_variant else verdicts.get(
            variant, "inconclusive",
        )
        rows.append({
            "base_model": base_model, "variant": variant, "model": identity["short"],
            "reference": variant == reference_variant,
            "quality_verdict": verdict,
            "quality_ranked": verdict in {"improved", "regressed"},
            "quality": _metric(values["quality"], reference["quality"], percentage_points=True),
            "throughput": _metric(values["throughput"], reference["throughput"]),
            "memory": _metric(values["memory"], reference["memory"]),
            "energy": _metric(values["energy"], reference["energy"]),
        })
    return {
        "artifact_type": "variant_comparison", "schema_version": VARIANT_COMPARISON_SCHEMA,
        "base_model": base_model, "reference_variant": reference_variant,
        "performance_section": performance_section, "case": case,
        "accuracy_section": accuracy_section, "variants": rows,
    }
