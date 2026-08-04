import json

import pytest

from scripts.setup.setup_progress import (
    finish_setup_progress,
    progress_status_text,
    read_progress_status,
    start_setup_progress,
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
        lambda command: launched.append(command) or process,
    )
    returned_process, returned_path = start_setup_progress()
    assert returned_process is process
    assert returned_path == status_path
    assert read_progress_status(status_path) == "running"
    assert launched[0][1:3] == ["-m", "scripts.setup.setup_progress"]


def test_start_setup_progress_removes_status_file_when_window_cannot_launch(monkeypatch, tmp_path):
    status_path = tmp_path / "progress.json"
    monkeypatch.setattr("scripts.setup.setup_progress.tempfile.mkstemp", lambda **_: (5, str(status_path)))
    monkeypatch.setattr("scripts.setup.setup_progress.os.close", lambda _handle: None)
    monkeypatch.setattr(
        "scripts.setup.setup_progress.subprocess.Popen",
        lambda _command: (_ for _ in ()).throw(OSError("cannot launch")),
    )
    with pytest.raises(OSError, match="cannot launch"):
        start_setup_progress()
    assert not status_path.exists()
