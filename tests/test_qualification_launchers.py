from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_unix_launcher_bootstraps_then_previews_by_default():
    text = (ROOT / "run_qualification.sh").read_text()
    assert 'python" -m pip install -r' in text
    assert "qualification_recipe" in text
    assert 'if [ "$EXECUTE" = "--execute" ]' in text


def test_windows_launcher_bootstraps_then_previews_by_default():
    text = (ROOT / "run_qualification.bat").read_text()
    assert "py -3 -m venv bench-env" in text
    assert "qualification_recipe" in text
    assert 'if "%EXECUTE%"=="--execute"' in text
