import hashlib
import json
from pathlib import Path

from decision_report import (
    build_report_model, render_html, write_html_report, write_pdf_report,
)
from decision_report_cli import main


FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_report_model_uses_explicit_evidence_without_composite_score():
    result = load("results_v4_1_complete.json")
    result["mcq"]["golden"] = {"accuracy_pct": 75.0, "correct": 3, "total": 4}
    model = build_report_model(result)
    assert model.title == "Local AI Bench Decision Report - commercial-golden-system"
    assert model.readiness == "COMPLETE EVIDENCE"
    assert model.performance == (
        ("Single-shot LLM", "golden", "2K", "50.00", "0.250"),
        ("Conversation", "golden", "0K", "48.00", "0.100"),
    )
    assert model.accuracy == (("MCQ", "golden", "75.0%", "3 / 4"),)
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
    assert main([str(result_path), "--html", str(html_path), "--pdf", str(pdf_path)]) == 0
    assert html_path.exists() and pdf_path.exists()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"metric": NaN}', encoding="utf-8")
    assert main([str(invalid), "--html", str(tmp_path / "bad.html")]) == 1
