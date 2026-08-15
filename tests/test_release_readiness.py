from scripts.release import release_readiness


def complete_evidence():
    evidence = {
        name: {
            "status": "passed", "approved_by": "release owner", "approved_at": "2026-08-04",
            "evidence": [f"records/{name}.json"],
        }
        for name in release_readiness.REQUIRED_EXTERNAL_GATES
    }
    evidence["telemetry_qualification"] = {
        mode: {
            "status": "passed", "protocol": "paired_observer_v1", "interval_sec": 0.5,
            "trial_pairs": 20, "sources": [f"{mode}-source"],
            "platform_classes": ["discrete-gpu"], "approved_by": "release owner",
            "approved_at": "2026-08-15", "evidence": [f"records/{mode}.json"],
        }
        for mode in release_readiness.REQUIRED_TELEMETRY_MODES
    }
    return evidence


def test_readiness_reports_each_unresolved_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(release_readiness, "frontend_option_gaps", lambda: ["--future"])
    monkeypatch.setattr(release_readiness, "model_catalog", lambda: [
        {"id": "model:one", "license": {"status": "unverified"}},
    ])
    monkeypatch.setattr(release_readiness, "HARDWARE_CATALOG", [
        {"id": "hardware:one", "qualification": "unqualified"},
    ])
    monkeypatch.setattr(release_readiness, "generate_sbom", lambda root: {"packages": [
        {"ecosystem": "pypi", "name": "unknown", "license": "NOASSERTION"},
        {"ecosystem": "npm", "name": "known", "license": "MIT"},
    ]})
    result = release_readiness.evaluate_release_readiness(tmp_path)
    checks = {check["name"]: check for check in result["checks"]}
    assert result["ready"] is False
    assert checks["frontend_option_coverage"]["items"] == ["--future"]
    assert checks["model_license_review"]["items"] == ["model:one"]
    assert checks["hardware_qualification"]["items"] == ["hardware:one"]
    assert checks["dependency_license_review"]["items"] == ["pypi:unknown"]
    assert checks["signed_installers"]["items"] == ["reviewed external evidence required"]
    assert checks["telemetry_source_qualification"]["items"] == [
        "combined", "memory", "power", "temperature",
    ]


def test_readiness_passes_when_all_local_inputs_are_cleared(monkeypatch, tmp_path):
    monkeypatch.setattr(release_readiness, "frontend_option_gaps", lambda: [])
    monkeypatch.setattr(release_readiness, "model_catalog", lambda: [
        {"id": "model:one", "license": {"status": "verified"}},
    ])
    monkeypatch.setattr(release_readiness, "HARDWARE_CATALOG", [
        {"id": "hardware:one", "qualification": "qualified"},
    ])
    monkeypatch.setattr(release_readiness, "generate_sbom", lambda root: {"packages": [
        {"ecosystem": "npm", "name": "known", "license": "MIT"},
    ]})
    result = release_readiness.evaluate_release_readiness(tmp_path, complete_evidence())
    assert result["ready"] is True
    assert all(check["passed"] for check in result["checks"])


def test_readiness_never_passes_with_only_local_checks_cleared(monkeypatch, tmp_path):
    monkeypatch.setattr(release_readiness, "frontend_option_gaps", lambda: [])
    monkeypatch.setattr(release_readiness, "model_catalog", lambda: [])
    monkeypatch.setattr(release_readiness, "HARDWARE_CATALOG", [])
    monkeypatch.setattr(release_readiness, "generate_sbom", lambda root: {"packages": []})
    result = release_readiness.evaluate_release_readiness(tmp_path)
    assert result["ready"] is False
    assert all(not check["passed"] for check in result["checks"] if check["name"]
               in release_readiness.REQUIRED_EXTERNAL_GATES)


def test_external_evidence_requires_approval_identity_date_and_references():
    valid = complete_evidence()["signed_installers"]
    assert release_readiness._external_check("signed_installers", valid)["passed"] is True
    for field, value in (("status", "pending"), ("approved_by", ""),
                         ("approved_at", ""), ("evidence", [])):
        invalid = dict(valid, **{field: value})
        assert release_readiness._external_check("signed_installers", invalid)["passed"] is False


def test_telemetry_gate_requires_protocol_and_complete_mode_evidence():
    records = complete_evidence()["telemetry_qualification"]
    assert release_readiness._telemetry_qualification_check(records)["passed"] is True
    records["power"] = dict(records["power"], trial_pairs=19)
    check = release_readiness._telemetry_qualification_check(records)
    assert check["passed"] is False
    assert check["items"] == ["power"]
