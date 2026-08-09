import json

import pytest

from scripts.runtime.pause_control import (
    apply_pause_evidence, create_pause_control, pause_evidence, read_pause_state,
    wait_if_paused, write_pause_state,
)
from scripts.results.result_history import discover_results


def test_pause_control_round_trip_and_unique_paths(tmp_path):
    first = create_pause_control(tmp_path)
    second = create_pause_control(tmp_path)
    assert first != second
    assert read_pause_state(first) == "running"
    write_pause_state(first, "paused")
    assert read_pause_state(first) == "paused"


def test_pause_transitions_become_run_evidence(tmp_path):
    path = create_pause_control(tmp_path)
    initial = pause_evidence(path)
    assert initial is None
    write_pause_state(path, "paused", now=lambda: "2026-08-04T10:00:00.000+00:00")
    write_pause_state(path, "running", now=lambda: "2026-08-04T10:05:00.000+00:00")
    evidence = pause_evidence(path)
    assert evidence is not None
    assert evidence["pause_requests"] == 1
    assert evidence["control_transitions"][-2:] == [
        {"state": "paused", "at": "2026-08-04T10:00:00.000+00:00"},
        {"state": "running", "at": "2026-08-04T10:05:00.000+00:00"},
    ]
    run = {}
    assert apply_pause_evidence(
        run, environ={"LOCAL_AI_BENCH_PAUSE_CONTROL": str(path)},
    ) is True
    assert run["pause"] == evidence


def test_default_pause_control_never_pollutes_result_history(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.runtime.pause_control.tempfile.gettempdir", lambda: str(tmp_path / "temp"))
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


def test_wait_treats_lost_or_malformed_control_as_running(tmp_path):
    missing = tmp_path / "missing.json"
    assert wait_if_paused(
        environ={"LOCAL_AI_BENCH_PAUSE_CONTROL": str(missing)}, sleep=lambda _: None,
    ) is False
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not json", encoding="utf-8")
    assert wait_if_paused(
        environ={"LOCAL_AI_BENCH_PAUSE_CONTROL": str(malformed)}, sleep=lambda _: None,
    ) is False


@pytest.mark.parametrize("value", [
    {"schema_version": 3, "state": "running"},
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
