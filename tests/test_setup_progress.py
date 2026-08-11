import json
import subprocess

import pytest

from scripts.setup.setup_progress import (
    finish_setup_progress,
    progress_status_text,
    read_progress_status,
    start_setup_progress,
    process_is_running,
)


def test_progress_status_reader_tolerates_missing_malformed_and_unknown_files(tmp_path):
    path = tmp_path / "progress.json"
    assert read_progress_status(path) == "running"
    path.write_text("not json")
    assert read_progress_status(path) == "running"
    path.write_text(json.dumps({"status": "future-value"}))
    assert read_progress_status(path) == "running"


@pytest.mark.parametrize("status", ["complete", "action_items", "stopped"])
def test_finish_setup_progress_records_terminal_status(tmp_path, status):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"status": "running"}))
    finish_setup_progress(path, status)
    assert read_progress_status(path) == status
    assert progress_status_text(status)


def test_finish_setup_progress_does_not_recreate_closed_window_status_file(tmp_path):
    path = tmp_path / "closed.json"
    finish_setup_progress(path, "complete")
    assert not path.exists()


def test_finish_setup_progress_rejects_unknown_status(tmp_path):
    with pytest.raises(ValueError, match="Unknown setup progress status"):
        finish_setup_progress(tmp_path / "progress.json", "unknown")


def test_start_setup_progress_initializes_status_and_launches_module(monkeypatch, tmp_path):
    status_path = tmp_path / "progress.json"
    monkeypatch.setattr("scripts.setup.setup_progress.tempfile.mkstemp", lambda **_: (5, str(status_path)))
    monkeypatch.setattr("scripts.setup.setup_progress.os.close", lambda _handle: None)
    launched = []
    process = object()
    monkeypatch.setattr(
        "scripts.setup.setup_progress.subprocess.Popen",
        lambda command, **kwargs: launched.append((command, kwargs)) or process,
    )
    returned_process, returned_path = start_setup_progress()
    assert returned_process is process
    assert returned_path == status_path
    assert read_progress_status(status_path) == "running"
    assert launched[0][0][1:3] == ["-m", "scripts.setup.setup_progress"]


def test_start_setup_progress_uses_separate_process_group_on_windows(monkeypatch, tmp_path):
    status_path = tmp_path / "progress.json"
    monkeypatch.setattr("scripts.setup.setup_progress.tempfile.mkstemp", lambda **_: (5, str(status_path)))
    monkeypatch.setattr("scripts.setup.setup_progress.os.close", lambda _handle: None)
    monkeypatch.setattr("scripts.setup.setup_progress.IS_WINDOWS", True)
    monkeypatch.setattr("scripts.setup.setup_progress.WINDOWS_NEW_PROCESS_GROUP", 512)
    launched = []
    monkeypatch.setattr(
        "scripts.setup.setup_progress.subprocess.Popen",
        lambda command, **kwargs: launched.append((command, kwargs)) or object(),
    )
    start_setup_progress()
    assert launched[0][1]["creationflags"] == 512


def test_process_is_running_uses_windows_probe_instead_of_os_kill(monkeypatch):
    monkeypatch.setattr("scripts.setup.setup_progress.IS_WINDOWS", True)
    monkeypatch.setattr("scripts.setup.setup_progress.windows_process_is_running", lambda pid: pid == 42)
    monkeypatch.setattr(
        "scripts.setup.setup_progress.os.kill",
        lambda *_args: pytest.fail("os.kill is unsafe as a Windows liveness probe"),
    )
    assert process_is_running(42)
    assert not process_is_running(41)


def test_start_setup_progress_removes_status_file_when_window_cannot_launch(monkeypatch, tmp_path):
    status_path = tmp_path / "progress.json"
    monkeypatch.setattr("scripts.setup.setup_progress.tempfile.mkstemp", lambda **_: (5, str(status_path)))
    monkeypatch.setattr("scripts.setup.setup_progress.os.close", lambda _handle: None)
    monkeypatch.setattr(
        "scripts.setup.setup_progress.subprocess.Popen",
        lambda _command, **_kwargs: (_ for _ in ()).throw(OSError("cannot launch")),
    )
    with pytest.raises(OSError, match="cannot launch"):
        start_setup_progress()
    assert not status_path.exists()
