"""Generate a deterministic SPDX-style dependency inventory."""

import json
import sys
from pathlib import Path


def generate_sbom(repo_root):
    """Build dependency records from committed Python and npm manifests."""
    root = Path(repo_root)
    packages = _python_packages(root / "requirements.txt", "runtime")
    packages += _python_packages(root / "tests" / "requirements.txt", "development")
    packages += _npm_packages(root / "dashboard" / "package-lock.json")
    packages.sort(key=lambda item: (item["ecosystem"], item["name"], item.get("version") or ""))
    return {"format": "local-ai-bench-sbom", "schema_version": 1, "packages": packages}


def write_sbom(repo_root, output_path):
    """Write canonical JSON so identical manifests produce identical output."""
    content = json.dumps(generate_sbom(repo_root), indent=2, sort_keys=True) + "\n"
    Path(output_path).write_text(content, encoding="utf-8")


def _python_packages(path, scope):
    packages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        name = requirement.split("[", 1)[0].split("=", 1)[0].strip()
        version = requirement.split("==", 1)[1] if "==" in requirement else None
        packages.append({
            "ecosystem": "pypi", "name": name, "version": version,
            "requirement": requirement, "scope": scope, "license": "NOASSERTION",
        })
    return packages


def _npm_packages(path):
    lock = json.loads(path.read_text(encoding="utf-8"))
    packages = []
    for location, value in lock["packages"].items():
        if not location.startswith("node_modules/"):
            continue
        packages.append({
            "ecosystem": "npm", "name": location.removeprefix("node_modules/"),
            "version": value.get("version"), "scope": "development" if value.get("dev") else "runtime",
            "license": value.get("license", "NOASSERTION"), "resolved": value.get("resolved"),
            "integrity": value.get("integrity"),
        })
    return packages


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/sbom.py OUTPUT.json")
    write_sbom(Path(__file__).resolve().parents[1], sys.argv[1])
