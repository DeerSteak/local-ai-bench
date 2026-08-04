from benchmark import checkpoint_terminal_exception


def test_interrupted_exception_checkpoints_pending_data_without_relabeling():
    results = {"run": {"status": "interrupted", "reason": "signal", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}, "llm": {"model": {"2K": {"tps": 10}}}}
    saved = []
    checkpoint_terminal_exception(results, SystemExit(), lambda label: saved.append((label, results.copy())))
    assert results["run"]["status"] == "interrupted"
    assert saved[0][0] == "run interrupted"
    assert saved[0][1]["llm"]["model"]["2K"]["tps"] == 10
    assert results["run"]["stages"]["llm"]["status"] == "interrupted"


def test_unhandled_exception_marks_run_failed():
    results = {"run": {"status": "running", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}}
    labels = []
    checkpoint_terminal_exception(results, RuntimeError(), labels.append)
    assert results["run"]["status"] == "failed"
    assert results["run"]["reason"] == "RuntimeError"
    assert labels == ["run failed"]
    assert results["run"]["stages"]["llm"]["status"] == "failed"


def test_nonfinite_exception_uses_specific_failure_reason():
    results = {"run": {"status": "running", "stages": {
        "llm": {"status": "running", "finished_at": None},
    }}}
    checkpoint_terminal_exception(
        results, ValueError("non-finite numeric value at $.llm.m.tps"), lambda label: None,
    )
    assert results["run"]["reason"] == "invalid_numeric_value"
    assert results["run"]["stages"]["llm"]["reason"] == "invalid_numeric_value"
