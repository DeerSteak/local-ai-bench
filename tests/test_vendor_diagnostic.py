import json

from vendor_diagnostic import (
    build_vendor_diagnostic, verify_vendor_diagnostic, write_vendor_diagnostic,
)
from vendor_diagnostic_cli import main


def result(hostname, tps, *, invalid=False):
    case = {
        "tps_mean": tps, "ttft_mean_sec": 0.2, "valid_runs": 2,
        "valid_samples": [{"tokens_per_sec": tps}],
        "invalid_runs": ([{"run": 3, "errors": ["decode_rate_mismatch"]}] if invalid else []),
    }
    return {
        "version": "4.1", "engine": "llamacpp", "profile": {"hostname": hostname},
        "run": {"plan": {"effective_config": {
            "methodology_profile": "neutral-v1", "runs": 3, "offline": True,
        }}},
        "llm": {"model": {"2K": case}},
    }


def test_diagnostic_identifies_first_measurement_and_preserves_raw_invalidity():
    baseline = result("A", 50.0)
    candidate = result("B", 45.0, invalid=True)
    diagnostic = build_vendor_diagnostic(baseline, candidate)
    assert diagnostic["first_divergence"]["metric"] == "llm/model/2K/tps_mean"
    assert diagnostic["raw_evidence"]["candidate"]["valid_samples"]
    assert diagnostic["invalidity"]["candidate"] == [
        {"run": 3, "errors": ["decode_rate_mismatch"]},
    ]
    assert verify_vendor_diagnostic(diagnostic, baseline, candidate)


def test_diagnostic_prioritizes_incompatible_identity_over_numeric_delta():
    baseline = result("A", 50.0)
    candidate = result("B", 45.0)
    candidate["run"]["plan"]["effective_config"]["runs"] = 5
    diagnostic = build_vendor_diagnostic(baseline, candidate)
    assert diagnostic["first_divergence"] == {
        "kind": "incompatible_identity", "fields": ["effective_config"],
    }


def test_diagnostic_bytes_are_deterministic_and_source_verified(tmp_path):
    baseline, candidate = result("A", 50.0), result("B", 45.0)
    baseline_path, candidate_path = tmp_path / "base.json", tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    first, second = tmp_path / "first.labdiag", tmp_path / "second.labdiag"
    write_vendor_diagnostic(baseline_path, candidate_path, first)
    write_vendor_diagnostic(baseline_path, candidate_path, second)
    assert first.read_bytes() == second.read_bytes()
    assert main(["verify", str(first), str(baseline_path), str(candidate_path)]) == 0
    candidate["profile"]["hostname"] = "changed"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    assert main(["verify", str(first), str(baseline_path), str(candidate_path)]) == 1


def test_cli_requires_metadata_review_before_writing(tmp_path):
    baseline_path, candidate_path = tmp_path / "base.json", tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(result("A", 50.0)), encoding="utf-8")
    candidate_path.write_text(json.dumps(result("B", 45.0)), encoding="utf-8")
    output = tmp_path / "diagnostic.labdiag"
    assert main(["create", str(baseline_path), str(candidate_path), str(output)]) == 1
    assert not output.exists()
    assert main([
        "create", str(baseline_path), str(candidate_path), str(output), "--reviewed-metadata",
    ]) == 0
    assert output.exists()
