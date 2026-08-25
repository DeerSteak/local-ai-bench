import json
import zipfile

import pytest

from scripts.results.workspace_export import (
    export_workspace_bundle, resolve_workspace_results, verify_workspace_bundle,
    write_workspace_reports,
)
from scripts.results.workspace_selection import build_workspace_selection
from scripts.results.workspace_export_cli import main


FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "results_v4_1_complete.json"


def copies(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(FIXTURE.read_bytes())
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["profile"]["hostname"] = "second"
    second.write_text(json.dumps(value), encoding="utf-8")
    return first, second


def test_workspace_bundle_is_deterministic_and_retains_selection(tmp_path):
    first, second = copies(tmp_path)
    selection = build_workspace_selection([first, second], baseline_path=second)
    one, two = tmp_path / "one.labworkspace", tmp_path / "two.labworkspace"
    export_workspace_bundle(selection, [second, first], one)
    export_workspace_bundle(selection, [first, second], two)
    assert one.read_bytes() == two.read_bytes()
    verified = verify_workspace_bundle(one)
    assert verified["selection"] == selection
    assert [result["profile"]["hostname"] for result in verified["results"]] == [
        "commercial-golden-system", "second",
    ]


def test_workspace_export_rejects_changed_or_duplicate_candidates(tmp_path):
    first, second = copies(tmp_path)
    selection = build_workspace_selection([first, second])
    second.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        resolve_workspace_results(selection, [first, second])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="distinct"):
        resolve_workspace_results(build_workspace_selection([first]), [first, duplicate])


def test_workspace_report_uses_baseline_and_records_full_selection(tmp_path):
    first, second = copies(tmp_path)
    selection = build_workspace_selection([first, second], baseline_path=second)
    html = tmp_path / "report.html"
    assert write_workspace_reports(selection, [first, second], html_path=html) == [html]
    rendered = html.read_text(encoding="utf-8")
    assert "Local AI Bench Decision Report - second" in rendered
    assert "Workspace selection" in rendered
    assert selection["results"][0]["sha256"] in rendered
    assert selection["results"][1]["sha256"] in rendered


def test_workspace_bundle_rejects_tampered_result(tmp_path):
    first, _ = copies(tmp_path)
    selection = build_workspace_selection([first])
    bundle = tmp_path / "bundle.labworkspace"
    export_workspace_bundle(selection, [first], bundle)
    tampered = tmp_path / "tampered.labworkspace"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "results/0.json":
                data += b" "
            target.writestr(info, data)
    with pytest.raises(ValueError, match="integrity check"):
        verify_workspace_bundle(tampered)


def test_workspace_export_cli_generates_bundle_and_report(tmp_path):
    first, _ = copies(tmp_path)
    selection = build_workspace_selection([first])
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    bundle, report = tmp_path / "workspace.labworkspace", tmp_path / "report.html"
    assert main([
        str(selection_path), "--result", str(first),
        "--bundle", str(bundle), "--html", str(report),
    ]) == 0
    assert bundle.exists() and report.exists()
    assert verify_workspace_bundle(bundle)["selection"] == selection
