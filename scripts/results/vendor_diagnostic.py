"""Deterministic vendor-engineer discrepancy evidence from two local results."""

import copy
from pathlib import Path

from scripts.results.canonical_json import sha256_json
from scripts.results.outbound_metadata import outbound_metadata_preview
from scripts.results.result_history import compare_results, load_result
from scripts.results.result_store import as_dict, atomic_write_json, validate_json_data


DIAGNOSTIC_SCHEMA_VERSION = 1


def _source_digest(result: dict) -> str:
    return sha256_json(result)


def _run_plan(result: dict):
    run = as_dict(result.get("run"))
    return copy.deepcopy(run.get("plan"))


def _evidence_for_metric(result: dict, metric_key: str):
    parts = metric_key.split("/")
    section = result.get(parts[0])
    if not isinstance(section, dict) or len(parts) < 3:
        return None
    model = section.get(parts[1])
    if not isinstance(model, dict):
        return None
    if parts[0] == "images" and len(parts) == 4:
        resolutions = model.get("resolutions")
        return copy.deepcopy(resolutions.get(parts[2])) if isinstance(resolutions, dict) else None
    if len(parts) == 4:
        return copy.deepcopy(model.get(parts[2]))
    return copy.deepcopy(model)


def _invalidity(evidence) -> list:
    if not isinstance(evidence, dict):
        return []
    records = copy.deepcopy(evidence.get("invalid_runs") or [])
    for key in ("skipped", "skip_reason", "timed_out", "timed_out_at", "crashed"):
        if key in evidence:
            records.append({key: copy.deepcopy(evidence[key])})
    return records


def build_vendor_diagnostic(baseline: dict, candidate: dict) -> dict:
    validate_json_data(baseline)
    validate_json_data(candidate)
    comparison = compare_results(baseline, candidate)
    divergent = next((
        row for row in comparison["rows"]
        if row["baseline"] != row["candidate"]
    ), None)
    baseline_evidence = _evidence_for_metric(baseline, divergent["metric"]) if divergent else None
    candidate_evidence = _evidence_for_metric(candidate, divergent["metric"]) if divergent else None
    first_divergence = (
        {"kind": "incompatible_identity", "fields": comparison["incompatible_fields"]}
        if not comparison["compatible"] else
        ({"kind": "measurement", **divergent} if divergent else {"kind": "none"})
    )
    diagnostic = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "source_sha256": {"baseline": _source_digest(baseline), "candidate": _source_digest(candidate)},
        "outbound_metadata": {
            "baseline": [list(row) for row in outbound_metadata_preview(baseline)],
            "candidate": [list(row) for row in outbound_metadata_preview(candidate)],
        },
        "environment": {
            "baseline": copy.deepcopy(baseline.get("profile")),
            "candidate": copy.deepcopy(candidate.get("profile")),
        },
        "run_plan": {"baseline": _run_plan(baseline), "candidate": _run_plan(candidate)},
        "first_divergence": first_divergence,
        "raw_evidence": {"baseline": baseline_evidence, "candidate": candidate_evidence},
        "invalidity": {
            "baseline": _invalidity(baseline_evidence),
            "candidate": _invalidity(candidate_evidence),
        },
        "reproduction_steps": [
            "Verify both source SHA-256 digests against the retained result files.",
            "Confirm the run-plan, runtime, methodology profile, and effective configuration identities.",
            "Re-run the exact plans on the named systems without changing tuning or validity rules.",
            "Inspect the first divergent case's raw samples and invalidity before comparing aggregates.",
        ],
    }
    validate_json_data(diagnostic)
    return diagnostic


def write_vendor_diagnostic(baseline_path: Path, candidate_path: Path, output_path: Path) -> Path:
    diagnostic = build_vendor_diagnostic(
        load_result(baseline_path), load_result(candidate_path),
    )
    atomic_write_json(Path(output_path), diagnostic)
    return Path(output_path)


def verify_vendor_diagnostic(diagnostic: dict, baseline: dict, candidate: dict) -> bool:
    if diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        return False
    expected = diagnostic.get("source_sha256")
    return expected == {"baseline": _source_digest(baseline), "candidate": _source_digest(candidate)}
