"""Generate a deterministic SPDX-style dependency inventory."""

import json
import sys
from pathlib import Path


PYTHON_LICENSE_RECORDS: dict[tuple[str, str], tuple[str, str, str | None]] = {
    ("gguf", "0.19.0"): ("MIT", "https://pypi.org/project/gguf/0.19.0/", None),
    ("huggingface_hub", "1.24.0"): (
        "Apache-2.0", "https://pypi.org/project/huggingface-hub/1.24.0/", None,
    ),
    ("numpy", "2.4.6"): (
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        "https://pypi.org/project/numpy/2.4.6/", None,
    ),
    ("packaging", "26.2"): (
        "Apache-2.0 OR BSD-2-Clause", "https://pypi.org/project/packaging/26.2/", None,
    ),
    ("psutil", "7.2.2"): (
        "BSD-3-Clause", "https://pypi.org/project/psutil/7.2.2/", None,
    ),
    ("py7zr", "1.1.3"): (
        "LGPL-2.1-or-later", "https://pypi.org/project/py7zr/1.1.3/",
        "Preserve LGPL notices and satisfy source and modification obligations when distributed.",
    ),
    ("pytest", "9.1.1"): ("MIT", "https://pypi.org/project/pytest/9.1.1/", None),
    ("reportlab", "5.0.1"): (
        "BSD-3-Clause", "https://pypi.org/project/reportlab/5.0.1/", None,
    ),
    ("requests", "2.34.2"): (
        "Apache-2.0", "https://pypi.org/project/requests/2.34.2/", None,
    ),
    ("tqdm", "4.69.0"): (
        "MPL-2.0 AND MIT", "https://pypi.org/project/tqdm/4.69.0/",
        "Preserve notices; distributed modifications to MPL-covered files remain under MPL-2.0.",
    ),
}


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
        license_record = PYTHON_LICENSE_RECORDS.get((name, version)) if version else None
        license_id, resolved, review_note = license_record or ("NOASSERTION", None, None)
        record = {
            "ecosystem": "pypi", "name": name, "version": version,
            "requirement": requirement, "scope": scope, "license": license_id,
        }
        if resolved:
            record["resolved"] = resolved
        if review_note:
            record["review_note"] = review_note
        packages.append(record)
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
        raise SystemExit("usage: python -m scripts.release.sbom OUTPUT.json")
    write_sbom(Path(__file__).resolve().parents[2], sys.argv[1])
