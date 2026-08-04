from third_party_notices import generate_notices, write_notices


def sample_sbom():
    return {"packages": [
        {"ecosystem": "pypi", "name": "unknown", "version": None, "scope": "runtime",
         "license": "NOASSERTION"},
        {"ecosystem": "npm", "name": "known", "version": "1.2.3", "scope": "development",
         "license": "MIT", "resolved": "https://registry.example/known"},
    ]}


def test_notices_list_dependencies_and_explicit_license_blockers():
    output = generate_notices(sample_sbom())
    assert "Unresolved license records: **1**" in output
    assert "| npm | known | 1.2.3 | development | MIT |" in output
    assert "`pypi:unknown` has no reviewed license assertion" in output


def test_notices_are_deterministic_and_escape_table_content():
    sbom = sample_sbom()
    sbom["packages"][0]["name"] = "unknown|package"
    assert generate_notices(sbom) == generate_notices(sbom)
    assert "unknown\\|package" in generate_notices(sbom)


def test_write_notices_uses_repository_sbom(monkeypatch, tmp_path):
    output = tmp_path / "THIRD_PARTY_NOTICES.md"
    monkeypatch.setattr("third_party_notices.generate_sbom", lambda _root: sample_sbom())
    write_notices(tmp_path, output)
    assert output.read_text(encoding="utf-8") == generate_notices(sample_sbom())
