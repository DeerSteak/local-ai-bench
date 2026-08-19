from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_unix_launcher_bootstraps_then_previews_by_default():
    text = (ROOT / "run_qualification.sh").read_text()
    assert 'python" -m pip install --quiet -r' in text
    assert 'VENV="$ROOT/qualification-env"' in text
    assert "qualification_recipe" in text
    assert 'if [ "$EXECUTE" = "--execute" ]' in text
    assert "scripts.release.qualification_auto" in text
    assert 'bootstrap_qualification.sh" --execute' in text


def test_windows_launcher_bootstraps_then_previews_by_default():
    text = (ROOT / "run_qualification.bat").read_text()
    assert "py -3 -m venv qualification-env" in text
    assert "qualification_recipe" in text
    assert 'if "%EXECUTE%"=="--execute"' in text
    assert "scripts.release.qualification_auto" in text
    assert "bootstrap_qualification.bat --execute" in text


def test_system_bootstraps_are_preview_first_and_leave_drivers_alone():
    unix = (ROOT / "bootstrap_qualification.sh").read_text()
    windows = (ROOT / "bootstrap_qualification.bat").read_text()
    assert 'if [ "$MODE" != "--execute" ]' in unix
    assert "GPU drivers and CUDA/ROCm SDKs are intentionally not changed" in unix
    assert 'if not "%~1"=="--execute"' in windows
    assert "winget install --id Python.Python.3.12" in windows


def test_target_listing_does_not_require_installed_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "scripts.release.qualification_recipe", "--list-targets"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "macos-m5-pro-llamacpp-metal" in result.stdout
