import json

import pytest

from scripts.release.sbom import generate_sbom, write_sbom


def make_repo(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "requirements.txt").write_text("requests\nreportlab==4.4.10\n", encoding="utf-8")
    (tmp_path / "tests" / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (tmp_path / "dashboard" / "package-lock.json").write_text(json.dumps({"packages": {
        "": {"name": "app"},
        "node_modules/react": {"version": "19.2.4", "license": "MIT", "integrity": "sha512-x"},
        "node_modules/vitest": {"version": "4.1.10", "dev": True},
    }}), encoding="utf-8")
    return tmp_path


def test_sbom_inventory_covers_manifests_and_preserves_unknown_licenses(tmp_path):
    sbom = generate_sbom(make_repo(tmp_path))
    packages = {(item["ecosystem"], item["name"]): item for item in sbom["packages"]}
    assert packages[("pypi", "reportlab")]["version"] == "4.4.10"
    assert packages[("pypi", "requests")]["license"] == "NOASSERTION"
    assert packages[("npm", "react")]["license"] == "MIT"
    assert packages[("npm", "vitest")]["scope"] == "development"


def test_sbom_output_is_deterministic(tmp_path):
    root = make_repo(tmp_path / "repo")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_sbom(root, first)
    write_sbom(root, second)
    assert first.read_bytes() == second.read_bytes()


def test_sbom_uses_reviewed_license_only_for_exact_python_version(tmp_path):
    root = make_repo(tmp_path)
    (root / "requirements.txt").write_text(
        "requests==2.34.2\npy7zr==1.1.3\n", encoding="utf-8",
    )
    packages = {item["name"]: item for item in generate_sbom(root)["packages"]}
    assert packages["requests"]["license"] == "Apache-2.0"
    assert packages["requests"]["resolved"].endswith("/requests/2.34.2/")
    assert packages["py7zr"]["license"] == "LGPL-2.1-or-later"
    assert "LGPL" in packages["py7zr"]["review_note"]


@pytest.mark.parametrize("name,version,license_id,url_name", [
    ("huggingface_hub", "1.30.0", "Apache-2.0", "huggingface-hub"),
    ("packaging", "26.3", "Apache-2.0 OR BSD-2-Clause", "packaging"),
    ("tqdm", "4.70.0", "MPL-2.0 AND MIT", "tqdm"),
])
def test_updated_python_license_records_are_version_specific(
    tmp_path, name, version, license_id, url_name,
):
    root = make_repo(tmp_path)
    (root / "requirements.txt").write_text(f"{name}=={version}\n", encoding="utf-8")
    package = next(p for p in generate_sbom(root)["packages"] if p["name"] == name)
    assert package["license"] == license_id
    assert package["resolved"] == f"https://pypi.org/project/{url_name}/{version}/"
    if name == "tqdm":
        assert "MPL-2.0" in package["review_note"]
    (root / "requirements.txt").write_text(f"{name}==999.0.0\n", encoding="utf-8")
    unknown = next(p for p in generate_sbom(root)["packages"] if p["name"] == name)
    assert unknown["license"] == "NOASSERTION"
    assert "resolved" not in unknown
    assert "review_note" not in unknown
