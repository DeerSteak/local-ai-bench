import json

import pytest

from pause_control import create_pause_control, read_pause_state, wait_if_paused, write_pause_state
from result_history import discover_results


def test_pause_control_round_trip_and_unique_paths(tmp_path):
    first = create_pause_control(tmp_path)
    second = create_pause_control(tmp_path)
    assert first != second
    assert read_pause_state(first) == "running"
    write_pause_state(first, "paused")
    assert read_pause_state(first) == "paused"


def test_default_pause_control_never_pollutes_result_history(tmp_path, monkeypatch):
    monkeypatch.setattr("pause_control.tempfile.gettempdir", lambda: str(tmp_path / "temp"))
    results = tmp_path / "results"
    results.mkdir()
    path = create_pause_control()
    assert results not in path.parents
    assert discover_results(results) == ([], [])
    path.unlink()


def test_wait_blocks_until_cooperative_resume(tmp_path):
    path = create_pause_control(tmp_path)
    write_pause_state(path, "paused")
    calls = []

    def resume(_seconds):
        calls.append("sleep")
        write_pause_state(path, "running")

    assert wait_if_paused(
        environ={"LOCAL_AI_BENCH_PAUSE_CONTROL": str(path)}, sleep=resume,
    ) is True
    assert calls == ["sleep"]
    assert wait_if_paused(environ={}) is False


@pytest.mark.parametrize("value", [
    {"schema_version": 2, "state": "running"},
    {"schema_version": 1, "state": "stopped"},
    [],
])
def test_pause_control_rejects_malformed_or_unknown_state(tmp_path, value):
    path = tmp_path / "control.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="malformed"):
        read_pause_state(path)
    with pytest.raises(ValueError, match="invalid pause state"):
        write_pause_state(path, "stopped")
