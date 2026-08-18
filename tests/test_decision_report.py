import hashlib
import json
from pathlib import Path

import pytest

from scripts.results.decision_report import (
    build_report_model, render_html, report_output_paths, write_html_report, write_pdf_report,
)
from scripts.results.decision_report_cli import main
from scripts.results.recommendation import evaluate_recommendation


FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_report_model_uses_explicit_evidence_without_composite_score():
    result = load("results_v4_1_complete.json")
    result["mcq"]["golden"] = {"accuracy_pct": 75.0, "correct": 3, "total": 4}
    result["run"]["plan"] = {"effective_config": {
        "methodology_profile": "neutral-v1",
        "effective_optimizations": ["llamacpp:flash_attention=on"],
        "offline": True,
    }}
    model = build_report_model(result)
    assert model.title == "Local AI Bench Decision Report - commercial-golden-system"
    assert model.readiness == "COMPLETE EVIDENCE"
    assert model.performance == (
        ("Single-shot LLM", "golden", "2K", "50.00", "0.250"),
        ("Conversation", "golden", "0K", "48.00", "0.100"),
    )
    assert model.accuracy == (("MCQ", "golden", "75.0%", "3 / 4"),)
    assert ("Methodology profile", "neutral-v1") in model.metadata
    assert ("Offline mode", "Yes") in model.metadata
    assert model.optimizations == ("llamacpp:flash_attention=on",)
    assert model.evidence == (
        ("Single-shot LLM", "golden / 2K", "2", "0"),
        ("Conversation", "golden / 0K", "1", "0"),
    )


def test_invalid_only_case_requires_review_without_an_aggregate():
    result = load("results_v4_1_complete.json")
    result["llm"]["golden"]["8K"] = {
        "completed_runs": 1,
        "valid_runs": 0,
        "invalid_runs": [{"reason": "unrealistic_tokens_per_sec"}],
        "valid_samples": [],
    }
    model = build_report_model(result)
    assert model.readiness == "REVIEW REQUIRED"
    assert ("Single-shot LLM", "golden / 8K", "0", "1") in model.evidence
    assert not any(row[2] == "8K" for row in model.performance)


def test_interrupted_report_requires_review_and_preserves_partial_evidence():
    model = build_report_model(load("results_v4_1_interrupted.json"))
    assert model.readiness == "REVIEW REQUIRED"
    assert "interrupted" in model.readiness_detail
    assert model.performance


def test_html_is_self_contained_escaped_and_deterministic(tmp_path):
    result = load("results_v4_1_complete.json")
    result["profile"]["hostname"] = "System <private>"
    model = build_report_model(result)
    first = render_html(model)
    second = render_html(model)
    assert first == second
    assert "System &lt;private&gt;" in first
    assert "http://" not in first and "https://" not in first
    assert "<script" not in first
    output = write_html_report(result, tmp_path / "report.html")
    assert output.read_text(encoding="utf-8") == first


def test_pdf_bytes_are_deterministic_and_contain_no_external_asset(tmp_path):
    result = load("results_v4_1_complete.json")
    first = write_pdf_report(result, tmp_path / "one.pdf").read_bytes()
    second = write_pdf_report(result, tmp_path / "two.pdf").read_bytes()
    assert first.startswith(b"%PDF-")
    assert hashlib.sha256(first).digest() == hashlib.sha256(second).digest()
    assert b"http://" not in first and b"https://" not in first


def test_report_cli_writes_both_formats_and_rejects_nonfinite_result(tmp_path):
    result_path = FIXTURES / "results_v4_1_complete.json"
    html_path, pdf_path = tmp_path / "report.html", tmp_path / "report.pdf"
    assert main([str(result_path), "--html", str(html_path)]) == 1
    assert not html_path.exists()
    assert main([
        str(result_path), "--html", str(html_path), "--pdf", str(pdf_path),
        "--reviewed-metadata",
    ]) == 0
    assert html_path.exists() and pdf_path.exists()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"metric": NaN}', encoding="utf-8")
    assert main([str(invalid), "--html", str(tmp_path / "bad.html")]) == 1


def test_report_cli_applies_acceptance_policy(tmp_path):
    result = load("results_v4_1_complete.json")
    result["run"]["plan"] = {"effective_config": {"methodology_profile": "neutral-v1"}}
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    policy = {
        "schema_version": 1, "name": "Gate", "methodology_profile": "neutral-v1",
        "rules": [{
            "id": "throughput", "section": "llm", "model": "golden", "case": "2K",
            "metric": "tps_mean", "operator": "at_least", "threshold": 55.0,
            "minimum_evidence": 2,
        }],
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    report_path = tmp_path / "report.html"
    assert main([
        str(result_path), "--html", str(report_path), "--policy", str(policy_path),
        "--reviewed-metadata",
    ]) == 0
    assert "Acceptance decision: REJECTED" in report_path.read_text(encoding="utf-8")


def test_report_output_paths_create_adjacent_html_and_pdf():
    assert report_output_paths(Path("results/vendor.report")) == (
        Path("results/vendor.html"), Path("results/vendor.pdf"),
    )


def test_report_renders_explicit_acceptance_without_changing_evidence_readiness():
    result = load("results_v4_1_complete.json")
    result["run"]["plan"] = {"effective_config": {"methodology_profile": "neutral-v1"}}
    policy = {
        "schema_version": 1, "name": "Gate", "methodology_profile": "neutral-v1",
        "rules": [{
            "id": "throughput", "section": "llm", "model": "golden", "case": "2K",
            "metric": "tps_mean", "operator": "at_least", "threshold": 55.0,
            "minimum_evidence": 2,
        }],
    }
    model = build_report_model(result, policy)
    assert model.readiness == "COMPLETE EVIDENCE"
    assert model.acceptance_decision == "REJECTED"
    assert model.acceptance == (("throughput", "fail", "50.0", "55.0", "2"),)
    assert "Acceptance decision: REJECTED" in render_html(model)


def test_report_renders_authoritative_recommendation_without_reevaluating_it():
    result = load("results_v4_1_complete.json")
    recommendation = evaluate_recommendation(result, {
        "workload": "llm", "case": "2K", "primary_objective": "throughput",
        "minimum_throughput": 40,
    })
    model = build_report_model(result, recommendation=recommendation)
    assert model.recommendation_verdict == "RECOMMENDED"
    assert model.recommendation_candidates[0] == (
        "Recommended", "golden", "50.0 tokens_per_second", "llm/golden/2K/tps_mean",
    )
    assert "Recommendation: RECOMMENDED" in render_html(model)


def test_report_rejects_recommendation_for_a_different_source_result():
    result = load("results_v4_1_complete.json")
    other = load("results_v4_1_complete.json")
    other["profile"]["hostname"] = "other"
    recommendation = evaluate_recommendation(other, {
        "workload": "llm", "case": "2K", "primary_objective": "throughput",
    })
    with pytest.raises(ValueError, match="does not cite this result"):
        build_report_model(result, recommendation=recommendation)


def test_report_cli_renders_recommendation_after_outbound_metadata_review(tmp_path):
    result = load("results_v4_1_complete.json")
    result_path = tmp_path / "result.json"
    artifact_path = tmp_path / "recommendation.json"
    report_path = tmp_path / "report.html"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    artifact_path.write_text(json.dumps(evaluate_recommendation(result, {
        "workload": "llm", "case": "2K", "primary_objective": "throughput",
    })), encoding="utf-8")
    assert main([
        str(result_path), "--html", str(report_path), "--reviewed-metadata",
        "--system-alias", "Reviewed system", "--recommendation", str(artifact_path),
    ]) == 0
    rendered = report_path.read_text(encoding="utf-8")
    assert "Recommendation: RECOMMENDED" in rendered
    assert "Reviewed system" in rendered
