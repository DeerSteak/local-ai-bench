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
    assert captured == ["-m", "scripts.app.benchmark_launcher", "--ui", "auto"]


def test_shell_wrapper_explicit_interface_launches_dispatcher(tmp_path):
    wrapper = make_isolated_shell_wrapper(tmp_path)
    result, captured = run_isolated_wrapper(wrapper, tmp_path, "--interface", "terminal")
    assert result.returncode == 0
    assert captured == ["-m", "scripts.app.benchmark_launcher", "--interface", "terminal"]


def test_shell_wrapper_ui_none_routes_through_dispatcher(tmp_path):
    wrapper = make_isolated_shell_wrapper(tmp_path)
    result, captured = run_isolated_wrapper(
        wrapper, tmp_path, "--ui", "none", "--tests", "llm",
    )
    assert result.returncode == 0
    assert captured == [
        "-m", "scripts.app.benchmark_launcher", "--ui", "none", "--tests", "llm",
    ]


def test_shell_wrapper_arguments_bypass_frontend_and_preserve_spaces(tmp_path):
    wrapper = make_isolated_shell_wrapper(tmp_path)
    result, captured = run_isolated_wrapper(
        wrapper, tmp_path, "--out", "my results file.json", "--tests", "llm",
    )
    assert result.returncode == 0
    assert captured == [
        "-m", "scripts.app.benchmark", "--out", "my results file.json", "--tests", "llm",
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
    assert 'python -m scripts.app.benchmark %*\nset "BENCH_EXIT_CODE=%errorlevel%"' in text
    assert ':frontend\npython -m scripts.app.benchmark_launcher --ui auto\nset "BENCH_EXIT_CODE=%errorlevel%"' in text
    assert 'if /i "%~1"=="--ui" goto frontend_with_args' in text
    assert 'if /i "%~1"=="--interface" goto frontend_with_args' in text
    assert ':frontend_with_args\npython -m scripts.app.benchmark_launcher %*' in text
    assert ':finish\nexit /b %BENCH_EXIT_CODE%' in text


def test_batch_wrapper_missing_venv_message_has_timestamp():
    text = (ROOT / "run_bench.bat").read_text()
    assert 'for /f "tokens=1 delims=." %%T in ("%TIME: =0%") do echo [%%T]' in text


def test_batch_wrapper_does_not_guess_how_cmd_was_invoked():
    text = (ROOT / "run_bench.bat").read_text()
    assert "CMDCMDLINE" not in text
    assert "pause" not in text.lower()


def test_double_click_benchmark_launchers_force_the_gui():
    command = (ROOT / "Run Local AI Bench.command").read_text()
    assert "bash run_bench.sh --ui gui" in command
    assert "osascript" not in command
    windows = (ROOT / "Run Local AI Bench.bat").read_text()
    assert "run_bench.bat --ui gui" in windows
    assert "pause" in windows.lower()
    assert 'exit /b %BENCH_EXIT_CODE%' in windows
    desktop = (ROOT / "Run Local AI Bench.desktop").read_text()
    assert "bash run_bench.sh --ui gui" in desktop
    assert os.access(ROOT / "Run Local AI Bench.command", os.X_OK)
    assert os.access(ROOT / "Run Local AI Bench.desktop", os.X_OK)


def test_dashboard_desktop_launchers_call_supported_wrappers():
    command = (ROOT / "Launch Local AI Bench Dashboard.command").read_text()
    desktop = (ROOT / "Launch Local AI Bench Dashboard.desktop").read_text()
    batch = (ROOT / "Launch Local AI Bench Dashboard.bat").read_text()
    assert "bash launch_dashboard.sh" in command
    assert "bash launch_dashboard.sh" in desktop
    assert "call launch_dashboard.bat" in batch


def test_dashboard_wrappers_use_the_authenticated_workspace_server():
    shell = (ROOT / "launch_dashboard.sh").read_text()
    batch = (ROOT / "launch_dashboard.bat").read_text()
    assert "-m scripts.app.workspace_server" in shell
    assert "-m scripts.app.workspace_server" in batch
    assert "-m scripts.app.dashboard_reuse" in shell
    assert "-m scripts.app.dashboard_reuse" in batch
    assert "run preview" not in shell
    assert "run preview" not in batch
    assert 'cd "$SCRIPT_DIR"' in shell
    assert 'pushd "%SCRIPT_DIR%"' in batch
    assert shell.index("npm run build") < shell.index("-m scripts.app.dashboard_reuse")
    assert batch.index("npm run build") < batch.index("-m scripts.app.dashboard_reuse")
