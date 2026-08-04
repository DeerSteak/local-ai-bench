from pathlib import Path

from result_bundle_cli import main


FIXTURE = Path(__file__).parent / "fixtures" / "results_v4_1_complete.json"


def test_bundle_cli_export_verify_and_import(tmp_path):
    bundle = tmp_path / "result.labresult"
    imported = tmp_path / "result.json"
    assert main(["export", str(FIXTURE), str(bundle)]) == 0
    assert main(["verify", str(bundle)]) == 0
    assert main(["import", str(bundle), str(imported)]) == 0
    assert imported.exists()


def test_bundle_cli_returns_failure_for_invalid_bundle(tmp_path):
    invalid = tmp_path / "invalid.labresult"
    invalid.write_text("not a zip", encoding="utf-8")
    assert main(["verify", str(invalid)]) == 1
