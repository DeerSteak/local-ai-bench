"""Deterministic self-contained HTML and PDF decision reports."""

import html
import json
from dataclasses import dataclass
from pathlib import Path

from result_store import validate_json_data


PERFORMANCE_SECTIONS = {
    "llm": "Single-shot LLM", "llm_conversation": "Conversation",
    "concurrency_tool": "Tool concurrency", "concurrency_chat": "Chat concurrency",
}


@dataclass(frozen=True)
class ReportModel:
    title: str
    metadata: tuple[tuple[str, str], ...]
    readiness: str
    readiness_detail: str
    coverage: tuple[tuple[str, str, str, str], ...]
    performance: tuple[tuple[str, str, str, str, str], ...]
    accuracy: tuple[tuple[str, str, str, str], ...]
    evidence: tuple[tuple[str, str, str, str], ...]
    optimizations: tuple[str, ...]
    acceptance_decision: str
    acceptance: tuple[tuple[str, str, str, str, str], ...]


def _text(value, fallback="Not recorded") -> str:
    return fallback if value is None or value == "" else str(value)


def _sample_counts(result: dict) -> tuple[int, int, int]:
    completed = result.get("completed_runs", result.get("n_runs", 0))
    valid = result.get("valid_runs", completed)
    invalid = len(result.get("invalid_runs") or [])
    return int(completed or 0), int(valid or 0), invalid


def build_report_model(result: dict, policy: dict | None = None) -> ReportModel:
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    hostname = _text(profile.get("hostname"), "Unnamed system")
    status = _text(run.get("status"), "legacy")
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
    settings = plan.get("effective_config") if isinstance(plan.get("effective_config"), dict) else {}
    metadata = (
        ("System", hostname), ("Application", _text(result.get("version"))),
        ("Engine", _text(result.get("engine") or run.get("engine"))),
        ("OS", _text(profile.get("os"))), ("Backend", _text(profile.get("backend"))),
        ("Memory", f"{profile['ram_gb']} GB" if profile.get("ram_gb") is not None else "Not recorded"),
        ("Run status", status), ("Plan ID", _text(run.get("plan_id"))),
        ("Methodology profile", _text(settings.get("methodology_profile"), "Legacy / not recorded")),
    )
    coverage = []
    stages = run.get("stages") if isinstance(run.get("stages"), dict) else {}
    for stage in run.get("stage_order", stages.keys()):
        state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        coverage.append((
            str(stage), _text(state.get("status"), "not recorded"),
            str(state.get("models_with_results") or 0),
            str((state.get("models_failed") or 0) + (state.get("models_skipped") or 0)),
        ))

    performance = []
    evidence = []
    invalid_total = 0
    completed_total = 0
    valid_total = 0
    for section, section_label in PERFORMANCE_SECTIONS.items():
        section_data = result.get(section)
        if not isinstance(section_data, dict):
            continue
        for model, model_data in section_data.items():
            if not isinstance(model_data, dict):
                continue
            for case_label, case in model_data.items():
                if not isinstance(case, dict):
                    continue
                has_sample_counts = any(key in case for key in (
                    "completed_runs", "n_runs", "valid_runs", "valid_samples", "invalid_runs",
                ))
                if not has_sample_counts:
                    continue
                completed, valid, invalid = _sample_counts(case)
                completed_total += completed
                valid_total += valid
                invalid_total += invalid
                evidence.append((
                    section_label, f"{model} / {case_label}", str(valid), str(invalid),
                ))
                if case.get("tps_mean") is not None:
                    performance.append((
                        section_label, str(model), str(case_label),
                        f"{case['tps_mean']:.2f}",
                        f"{case.get('client_ttft_mean_sec', case.get('ttft_mean_sec', 0)):.3f}",
                    ))

    accuracy = []
    for section in ("mcq", "math", "reasoning", "code", "tool"):
        section_data = result.get(section)
        if not isinstance(section_data, dict):
            continue
        for model, score in section_data.items():
            if isinstance(score, dict) and score.get("accuracy_pct") is not None:
                accuracy.append((
                    section.upper(), str(model), f"{score['accuracy_pct']:.1f}%",
                    f"{score.get('correct', 0)} / {score.get('total', 0)}",
                ))

    if status != "complete":
        readiness = "REVIEW REQUIRED"
        readiness_detail = f"Run status is {status}; partial evidence is preserved but not complete."
    elif invalid_total:
        readiness = "REVIEW REQUIRED"
        readiness_detail = (
            f"{invalid_total} of {completed_total} performance samples were excluded; "
            "review their recorded reasons before using this run for a decision."
        )
    else:
        readiness = "COMPLETE EVIDENCE"
        readiness_detail = (
            f"All {valid_total} recorded performance samples represented here were aggregate-eligible."
        )
    acceptance_decision = "Not evaluated"
    acceptance = ()
    if policy is not None:
        from acceptance_policy import evaluate_policy

        evaluation = evaluate_policy(result, policy)
        acceptance_decision = evaluation["decision"].upper()
        acceptance = tuple((
            item["id"], item["status"], _text(item["actual"]),
            _text(item["threshold"]), str(item["evidence"]),
        ) for item in evaluation["rules"])
    return ReportModel(
        title=f"Local AI Bench Decision Report - {hostname}", metadata=metadata,
        readiness=readiness, readiness_detail=readiness_detail,
        coverage=tuple(coverage), performance=tuple(performance), accuracy=tuple(accuracy),
        evidence=tuple(evidence), optimizations=tuple(settings.get("effective_optimizations") or ()),
        acceptance_decision=acceptance_decision, acceptance=acceptance,
    )


def _html_table(headers, rows) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(value))}</td>" for value in row
    ) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(model: ReportModel) -> str:
    metadata = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>"
        for label, value in model.metadata
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(model.title)}</title><style>
body{{font:15px/1.45 Arial,sans-serif;color:#17202a;max-width:1100px;margin:0 auto;padding:38px}}
h1{{font-size:30px;margin:0 0 6px}}h2{{margin-top:30px;border-bottom:2px solid #2457c5;padding-bottom:6px}}
.eyebrow{{color:#2457c5;letter-spacing:.16em;text-transform:uppercase;font-weight:700}}
.readiness{{border-left:6px solid #2457c5;background:#eef4ff;padding:16px 18px;margin:24px 0}}
.readiness strong{{display:block;font-size:18px}}dl{{display:grid;grid-template-columns:160px 1fr;gap:5px 14px}}
dt{{font-weight:700}}dd{{margin:0}}table{{width:100%;border-collapse:collapse;margin:10px 0 22px}}
th,td{{padding:8px 10px;border-bottom:1px solid #d7dde5;text-align:left}}th{{background:#eef2f7;font-size:12px;text-transform:uppercase}}
.limitations{{background:#f6f8fa;padding:15px 18px}}footer{{margin-top:34px;color:#667085;font-size:12px}}
@media print{{body{{padding:0}}h2{{break-after:avoid}}table{{break-inside:auto}}tr{{break-inside:avoid}}}}
</style></head><body><div class="eyebrow">Local AI Bench 4.1</div><h1>{html.escape(model.title)}</h1>
<div class="readiness"><strong>{html.escape(model.readiness)}</strong>{html.escape(model.readiness_detail)}</div>
<h2>Run identity</h2><dl>{metadata}</dl>
<h2>Coverage</h2>{_html_table(("Stage", "Status", "Models with results", "Skipped / failed"), model.coverage)}
<h2>Performance evidence</h2>{_html_table(("Workload", "Model", "Case", "Tokens/sec", "TTFT sec"), model.performance)}
<h2>Accuracy evidence</h2>{_html_table(("Workload", "Model", "Accuracy", "Correct"), model.accuracy)}
<h2>Sample validity</h2>{_html_table(("Workload", "Case", "Valid", "Excluded"), model.evidence)}
<h2>Effective optimizations</h2>{_html_table(("Recorded setting",), ((value,) for value in model.optimizations))}
<h2>Acceptance decision: {html.escape(model.acceptance_decision)}</h2>{_html_table(("Rule", "Status", "Actual", "Threshold", "Evidence"), model.acceptance)}
<h2>Interpretation limits</h2><div class="limitations">This report presents measured evidence, not a hidden composite score or a universal recommendation. Compare only compatible methodology, model artifacts, runtimes, cache semantics, and effective settings. Missing or invalid data is not zero. Review raw samples, exclusion reasons, and the verified result bundle before making a purchase, launch, or capacity decision.</div>
<footer>Generated deterministically from a Local AI Bench result. No external assets, scripts, telemetry, or network resources are embedded.</footer>
</body></html>"""


def write_html_report(result: dict, path: Path, policy: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(build_report_model(result, policy)), encoding="utf-8", newline="\n")
    return path


def report_output_paths(html_path: Path) -> tuple[Path, Path]:
    html_path = Path(html_path).with_suffix(".html")
    return html_path, html_path.with_suffix(".pdf")


def write_pdf_report(result: dict, path: Path, policy: dict | None = None) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    model = build_report_model(result, policy)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#17202a")
    styles["Heading2"].textColor = colors.HexColor("#2457c5")
    doc = SimpleDocTemplate(
        str(path), pagesize=letter, leftMargin=.55 * inch, rightMargin=.55 * inch,
        topMargin=.55 * inch, bottomMargin=.55 * inch, invariant=1,
        title=model.title, author="Local AI Bench",
    )
    story = [Paragraph("LOCAL AI BENCH 4.1", styles["Heading3"]),
             Paragraph(html.escape(model.title), styles["Title"]), Spacer(1, 10),
             Paragraph(
                 f"<b>{html.escape(model.readiness)}</b><br/>{html.escape(model.readiness_detail)}",
                 styles["BodyText"],
             )]

    def add_table(title, headers, rows, widths=None):
        display_rows = rows or (("No evidence recorded",) + ("",) * (len(headers) - 1),)
        values = [list(headers)] + [list(row) for row in display_rows]
        table = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17202a")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#c7d0dc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        section = [Spacer(1, 12), Paragraph(title, styles["Heading2"]), table]
        if len(display_rows) <= 10:
            story.append(KeepTogether(section))
        else:
            story.extend(section)

    add_table("Run identity", ("Field", "Value"), model.metadata, [1.45 * inch, 5.85 * inch])
    add_table("Coverage", ("Stage", "Status", "Models", "Skipped / failed"), model.coverage)
    add_table("Performance evidence", ("Workload", "Model", "Case", "Tokens/sec", "TTFT sec"), model.performance)
    add_table("Accuracy evidence", ("Workload", "Model", "Accuracy", "Correct"), model.accuracy)
    add_table("Sample validity", ("Workload", "Case", "Valid", "Excluded"), model.evidence)
    add_table("Effective optimizations", ("Recorded setting",), tuple((value,) for value in model.optimizations))
    add_table(
        f"Acceptance decision: {model.acceptance_decision}",
        ("Rule", "Status", "Actual", "Threshold", "Evidence"), model.acceptance,
    )
    story.extend([Spacer(1, 12), Paragraph("Interpretation limits", styles["Heading2"]), Paragraph(
        "This report presents measured evidence, not a hidden composite score or a universal recommendation. "
        "Compare only compatible methodology, model artifacts, runtimes, cache semantics, and effective settings. "
        "Missing or invalid data is not zero. Review raw samples, exclusion reasons, and the verified result bundle "
        "before making a purchase, launch, or capacity decision.", styles["BodyText"],
    )])
    doc.build(story)
    return path


def load_result(path: Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    return result
