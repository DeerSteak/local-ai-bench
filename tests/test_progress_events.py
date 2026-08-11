import json

import pytest

from scripts.app.progress_events import (
    PROGRESS_PREFIX, emit_model_finished, emit_progress, emit_result_saved, set_progress_engine,
)


def test_model_progress_preserves_labels_with_punctuation(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    emit_progress("model", "llm", "running", "Qwen: 4B / Q4_K_M")
    assert capsys.readouterr().out == (
        '::local-ai-bench-progress::{"kind":"model","stage":"llm",'
        '"status":"running","model":"Qwen: 4B / Q4_K_M"}\n'
    )


@pytest.mark.parametrize("exc,status", [
    (None, "complete"), (KeyboardInterrupt(), "interrupted"), (RuntimeError("boom"), "failed"),
])
def test_model_terminal_status_reflects_unwinding_exception(monkeypatch, capsys, exc, status):
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    try:
        try:
            if exc is not None:
                raise exc
        finally:
            emit_model_finished("llm", "Model")
    except BaseException:
        pass
    assert f'"status":"{status}"' in capsys.readouterr().out


def test_model_progress_reports_usable_saved_results(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    emit_model_finished("llm", "Model", {"2K": {"tps_mean": 12.5}})
    assert '"usable":true' in capsys.readouterr().out


def _emit(capsys, **kwargs):
    emit_progress(**kwargs)
    line = capsys.readouterr().out.strip()
    return json.loads(line[len(PROGRESS_PREFIX):]) if line else None


def test_events_carry_the_running_engine(monkeypatch, capsys):
    """Without this, a two-engine run's passes overwrite each other's progress rows."""
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    try:
        set_progress_engine("vllm")
        payload = _emit(capsys, kind="model", stage="llm", status="running", model="Gemma 3 1B")
        assert payload is not None and payload["engine"] == "vllm"
        assert (payload["stage"], payload["model"]) == ("llm", "Gemma 3 1B")

        set_progress_engine("llamacpp")
        stage_payload = _emit(capsys, kind="stage", stage="llm", status="complete")
        assert stage_payload is not None and stage_payload["engine"] == "llamacpp"
    finally:
        set_progress_engine(None)


def test_engine_is_omitted_when_unset(monkeypatch, capsys):
    """Single-engine callers and older consumers see the original payload shape."""
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    set_progress_engine(None)
    payload = _emit(capsys, kind="stage", stage="llm", status="running")
    assert payload is not None and "engine" not in payload


def test_result_saved_uses_private_structured_progress_channel(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    set_progress_engine(None)
    result = tmp_path / "results_workstation.json"
    emit_result_saved(result)
    payload = json.loads(capsys.readouterr().out.strip().removeprefix(PROGRESS_PREFIX))
    assert payload == {
        "kind": "result", "stage": "run", "status": "complete", "path": str(result),
    }
