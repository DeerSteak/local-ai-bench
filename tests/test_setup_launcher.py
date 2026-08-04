from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_macos_launcher_treats_gui_cancel_as_clean_exit():
    launcher = (ROOT / "Setup Local AI Bench.command").read_text()
    cancel_check = 'if [ "$status" -eq 10 ]'
    error_check = 'if [ "$status" -ne 0 ]'
    assert cancel_check in launcher
    assert launcher.index(cancel_check) < launcher.index(error_check)
    assert "close_terminal_tab.applescript" in launcher
    assert '"$LAUNCHER_TTY"' in launcher


def test_terminal_closer_targets_the_launching_tty():
    script = (ROOT / "scripts" / "close_terminal_tab.applescript").read_text()
    assert "tty of terminalTab is targetTTY" in script
    assert "delay 0.2" in script
    assert "close terminalWindow" in script
    assert 'keystroke "w" using command down' in script
    assert "error " not in script
    launcher = (ROOT / "Setup Local AI Bench.command").read_text()
    assert "nohup /usr/bin/osascript" in launcher
    assert "launchctl submit" not in launcher


def test_setup_wrapper_only_offers_benchmark_after_setup_check_succeeds():
    wrapper = (ROOT / "setup.sh").read_text()
    setup_call = '"$VENV_PYTHON" -m scripts.setup.setup_check "$@"'
    benchmark_prompt = 'Run the benchmark now? [y/N]'
    assert wrapper.index(setup_call) < wrapper.index(benchmark_prompt)
    assert "set -euo pipefail" in wrapper


def test_linux_desktop_launcher_is_repo_relative_and_forces_gui():
    launcher = (ROOT / "Setup Local AI Bench.desktop").read_text()
    assert launcher.startswith("[Desktop Entry]\n")
    assert "Type=Application" in launcher
    assert "Terminal=true" in launcher
    assert "%k" in launcher
    assert 'cd "$(dirname "$p")"' in launcher
    assert "bash setup.sh --interface gui" in launcher


def test_windows_launcher_forces_gui_and_treats_cancel_as_clean_exit():
    launcher = (ROOT / "Setup Local AI Bench.bat").read_text()
    wrapper = (ROOT / "setup.bat").read_text()
    assert 'cd /d "%~dp0"' in launcher
    assert "call setup.bat --interface gui" in launcher
    assert "if %SETUP_STATUS% equ 10 exit /b 0" in launcher
    assert "if %errorlevel% equ 10 exit /b 10" in wrapper
