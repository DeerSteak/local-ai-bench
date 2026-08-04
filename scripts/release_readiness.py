"""Local release-readiness checks with explicit blocking evidence."""

import json
import sys
from pathlib import Path

from benchmark_frontend import frontend_option_gaps
from catalogs import HARDWARE_CATALOG, model_catalog
from sbom import generate_sbom


def evaluate_release_readiness(repo_root):
    """Return deterministic checks; external qualification remains outside this preflight."""
    sbom = generate_sbom(repo_root)
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
    return {"schema_version": 1, "ready": all(check["passed"] for check in checks), "checks": checks}


def _check(name, passed, items):
    return {"name": name, "passed": bool(passed), "items": sorted(items)}


if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    result = evaluate_release_readiness(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["ready"] else 1)
