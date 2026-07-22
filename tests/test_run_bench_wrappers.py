import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_isolated_shell_wrapper(tmp_path):
    wrapper = tmp_path / "run_bench.sh"
    wrapper.write_text((ROOT / "run_bench.sh").read_text())
    wrapper.chmod(0o755)

    bin_dir = tmp_path / "bench-env" / "bin"
    bin_dir.mkdir(parents=True)
    activate = bin_dir / "activate"
    activate.write_text(f'export PATH="{bin_dir}:$PATH"\n')
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n"
        "exit \"${FAKE_EXIT_CODE:-0}\"\n"
    )
    fake_python.chmod(0o755)
    return wrapper


def run_isolated_wrapper(wrapper, tmp_path, *args, exit_code=0):
    capture = tmp_path / "captured.txt"
    env = {**os.environ, "CAPTURE_PATH": str(capture), "FAKE_EXIT_CODE": str(exit_code)}
    result = subprocess.run(
        ["bash", str(wrapper), *args], text=True, capture_output=True, env=env,
    )
    captured = capture.read_text().splitlines() if capture.exists() else []
    return result, captured


def test_shell_wrapper_zero_arguments_launches_frontend(tmp_path):
    wrapper = make_isolated_shell_wrapper(tmp_path)
    result, captured = run_isolated_wrapper(wrapper, tmp_path)
    assert result.returncode == 0
    assert captured == [str(tmp_path / "scripts" / "benchmark_frontend.py")]


def test_shell_wrapper_arguments_bypass_frontend_and_preserve_spaces(tmp_path):
    wrapper = make_isolated_shell_wrapper(tmp_path)
    result, captured = run_isolated_wrapper(
        wrapper, tmp_path, "--out", "my results file.json", "--tests", "llm",
    )
    assert result.returncode == 0
    assert captured == [
        str(tmp_path / "scripts" / "benchmark.py"),
        "--out", "my results file.json", "--tests", "llm",
    ]


def test_shell_wrapper_propagates_child_exit_code(tmp_path):
    wrapper = make_isolated_shell_wrapper(tmp_path)
    result, _ = run_isolated_wrapper(wrapper, tmp_path, "--help", exit_code=2)
    assert result.returncode == 2


def test_shell_wrapper_missing_venv_message_has_timestamp(tmp_path):
    wrapper = tmp_path / "run_bench.sh"
    wrapper.write_text((ROOT / "run_bench.sh").read_text())
    result = subprocess.run(["bash", str(wrapper)], text=True, capture_output=True)
    assert result.returncode == 1
    assert re.match(r"^\[\d{2}:\d{2}:\d{2}\] Virtual environment not found", result.stdout)


def test_batch_wrapper_uses_label_branches_and_preserves_exit_codes():
    text = (ROOT / "run_bench.bat").read_text()
    assert 'if "%~1"=="" goto frontend' in text
    assert 'python "%SCRIPT_DIR%scripts\\benchmark.py" %*\nset "BENCH_EXIT_CODE=%errorlevel%"' in text
    assert ':frontend\npython "%SCRIPT_DIR%scripts\\benchmark_frontend.py"\nset "BENCH_EXIT_CODE=%errorlevel%"' in text
    assert ':finish\nif defined PAUSE_ON_EXIT pause\nexit /b %BENCH_EXIT_CODE%' in text


def test_batch_wrapper_missing_venv_message_has_timestamp():
    text = (ROOT / "run_bench.bat").read_text()
    assert 'for /f "tokens=1 delims=." %%T in ("%TIME: =0%") do echo [%%T]' in text


def test_batch_wrapper_only_pauses_for_explorer_style_invocation():
    text = (ROOT / "run_bench.bat").read_text()
    assert 'set "PAUSE_ON_EXIT="' in text
    assert 'set CMDCMDLINE | %SystemRoot%\\System32\\findstr.exe /l /i /c:"%~f0" >nul' in text
    assert 'if not errorlevel 1 set "PAUSE_ON_EXIT=1"' in text
    assert text.count('if defined PAUSE_ON_EXIT pause') == 1
