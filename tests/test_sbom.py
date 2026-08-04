import json

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
