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
    assert "/usr/lib/wsl/lib" in text


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
    assert "python3.12-venv" not in unix
    assert "uv" in unix and "python install 3.12" in unix
    assert "libopenmpi-dev" in unix and "openmpi-devel" in unix
    assert 'if not "%~1"=="--execute"' in windows
    assert "winget install --id Python.Python.3.12" in windows


def test_posix_launcher_accepts_python_314_for_its_own_environment(tmp_path):
    python = tmp_path / "python3.14"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    command = (
        f'source "{ROOT / "qualification_python.sh"}"; qualification_python'
    )
    result = subprocess.run(
        ["/bin/bash", "-c", command], capture_output=True, text=True,
        env={"PATH": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    assert result.stdout.strip() == str(python)


def test_posix_python_selector_rejects_an_unsupported_interpreter(tmp_path):
    python = tmp_path / "python3"
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    command = f'source "{ROOT / "qualification_python.sh"}"; qualification_python'
    result = subprocess.run(
        ["/bin/bash", "-c", command], capture_output=True, text=True,
        env={"PATH": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert result.returncode != 0


def test_vllm_python_selector_does_not_mistake_python_314_for_312(tmp_path):
    python = tmp_path / "python3.12"
    python.write_text("#!/bin/sh\nexit 1\n")
    python.chmod(0o755)
    command = f'source "{ROOT / "qualification_python.sh"}"; qualification_python_312'
    result = subprocess.run(
        ["/bin/bash", "-c", command], capture_output=True, text=True,
        env={"PATH": str(tmp_path), "HOME": str(tmp_path)},
    )
    assert result.returncode != 0


def test_posix_qualification_scripts_have_valid_shell_syntax():
    for name in ("qualification_python.sh", "bootstrap_qualification.sh", "run_qualification.sh"):
        result = subprocess.run(["/bin/bash", "-n", ROOT / name], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_target_listing_does_not_require_installed_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "scripts.release.qualification_recipe", "--list-targets"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "macos-m5-pro-llamacpp-metal" in result.stdout
