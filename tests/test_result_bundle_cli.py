from pathlib import Path

from scripts.results.result_bundle_cli import main


FIXTURE = Path(__file__).parent / "fixtures" / "results_v4_1_complete.json"


def test_bundle_cli_export_verify_and_import(tmp_path):
    bundle = tmp_path / "result.labresult"
    imported = tmp_path / "result.json"
    assert main(["export", str(FIXTURE), str(bundle)]) == 1
    assert not bundle.exists()
    assert main(["export", str(FIXTURE), str(bundle), "--reviewed-metadata"]) == 0
    assert main(["verify", str(bundle), "--source-result", str(FIXTURE)]) == 0
    assert main(["import", str(bundle), str(imported)]) == 0
    assert imported.exists()


def test_bundle_cli_returns_failure_for_invalid_bundle(tmp_path):
    invalid = tmp_path / "invalid.labresult"
    invalid.write_text("not a zip", encoding="utf-8")
    assert main(["verify", str(invalid)]) == 1


def test_bundle_cli_rejects_mismatched_private_source(tmp_path):
    bundle = tmp_path / "result.labresult"
    assert main(["export", str(FIXTURE), str(bundle), "--reviewed-metadata"]) == 0
    other = tmp_path / "other.json"
    other.write_text('{"profile":{"hostname":"other"}}', encoding="utf-8")
    assert main(["verify", str(bundle), "--source-result", str(other)]) == 1
