import pytest

from progress_events import emit_model_finished, emit_progress


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
