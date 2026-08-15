"""Local release-readiness checks with explicit blocking evidence."""

import json
import sys
from pathlib import Path

from scripts.app.benchmark_frontend import frontend_option_gaps
from scripts.results.catalogs import HARDWARE_CATALOG, model_catalog
from scripts.release.sbom import generate_sbom


REQUIRED_EXTERNAL_GATES = (
    "signed_installers", "release_security_scans", "offline_platform_qualification",
    "clean_machine_lifecycle", "accessibility_and_usability", "legal_approval",
    "independent_security_assessment", "stable_release_approval",
)
REQUIRED_TELEMETRY_MODES = ("memory", "power", "temperature", "combined")


def evaluate_release_readiness(repo_root, evidence=None):
    """Return local checks plus explicitly approved external release evidence."""
    sbom = generate_sbom(repo_root)
    evidence = evidence or {}
    checks = [
        _check("frontend_option_coverage", not frontend_option_gaps(), frontend_option_gaps()),
        _check("model_license_review", False, [
            record["id"] for record in model_catalog()
            if record["license"]["status"] != "verified"
        ]),
        _check("hardware_qualification", False, [
            record["id"] for record in HARDWARE_CATALOG
            if record["qualification"] != "qualified"
        ]),
        _check("dependency_license_review", False, [
            f"{record['ecosystem']}:{record['name']}" for record in sbom["packages"]
            if record["license"] == "NOASSERTION"
        ]),
    ]
    for check in checks:
        check["passed"] = not check["items"]
    checks += [_external_check(name, evidence.get(name)) for name in REQUIRED_EXTERNAL_GATES]
    checks.append(_telemetry_qualification_check(evidence.get("telemetry_qualification")))
    return {"schema_version": 1, "ready": all(check["passed"] for check in checks), "checks": checks}


def _check(name, passed, items):
    return {"name": name, "passed": bool(passed), "items": sorted(items)}


def _external_check(name, record):
    valid = (
        isinstance(record, dict) and record.get("status") == "passed"
        and isinstance(record.get("approved_by"), str) and bool(record["approved_by"].strip())
        and isinstance(record.get("approved_at"), str) and bool(record["approved_at"].strip())
        and isinstance(record.get("evidence"), list) and bool(record["evidence"])
        and all(isinstance(item, str) and item.strip() for item in record["evidence"])
    )
    return {
        "name": name, "passed": bool(valid),
        "items": [] if valid else ["reviewed external evidence required"],
    }


def _telemetry_qualification_check(records):
    failures = []
    records = records if isinstance(records, dict) else {}
    for mode in REQUIRED_TELEMETRY_MODES:
        record = records.get(mode)
        valid = (
            isinstance(record, dict) and record.get("status") == "passed"
            and record.get("protocol") == "paired_observer_v1"
            and isinstance(record.get("interval_sec"), (int, float))
            and not isinstance(record.get("interval_sec"), bool) and record["interval_sec"] > 0
            and isinstance(record.get("trial_pairs"), int)
            and not isinstance(record.get("trial_pairs"), bool) and record["trial_pairs"] >= 20
            and isinstance(record.get("sources"), list) and bool(record["sources"])
            and all(isinstance(item, str) and item.strip() for item in record["sources"])
            and isinstance(record.get("platform_classes"), list)
            and bool(record["platform_classes"])
            and all(isinstance(item, str) and item.strip()
                    for item in record["platform_classes"])
            and _external_check(mode, record)["passed"]
        )
        if not valid:
            failures.append(mode)
    return _check("telemetry_source_qualification", not failures, failures)


if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[2]
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m scripts.release.release_readiness [EVIDENCE.json]")
    evidence = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")) if len(sys.argv) == 2 else None
    result = evaluate_release_readiness(root, evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["ready"] else 1)
