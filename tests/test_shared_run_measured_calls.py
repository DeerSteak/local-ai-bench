from typing import TYPE_CHECKING, cast

import requests
import pytest
from scripts.runtime import shared

from scripts.runtime.shared import (
    EngineBudgetExceeded,
    EngineLoopDetected,
    EngineTimeout,
    Shared,
    split_token_budget,
)
from scripts.runtime import config
from scripts.runtime.engines.base import GenerationMeasurement


class _FakeEngine:
    """Minimal double for the engine methods run_measured_calls calls;
    `recovers` controls whether a crash is treated as recoverable."""

    name = "fake"

    def __init__(self, recovers=True):
        self._recovers = recovers

    def is_connection_crash(self, e):
        return isinstance(e, requests.exceptions.ConnectionError) or "actively refused" in str(e).lower()

    def tail_log(self, n_lines=40):
        return "(fake engine log)"

    def wait_for_recovery(self, timeout=30):
        return self._recovers


if TYPE_CHECKING:
    from scripts.runtime.engines.base import InferenceEngine


def _engine(recovers=True) -> "InferenceEngine":
    """cast, not a subclass: the fake implements only the 3 methods this module's
    functions actually call on an engine, not the full InferenceEngine ABC."""
    return cast("InferenceEngine", _FakeEngine(recovers=recovers))


@pytest.mark.parametrize(("total", "expected"), [
    (1, (1, 0)),
    (2, (1, 1)),
    (3, (1, 2)),
    (5, (3, 2)),
    (8192, (4915, 3277)),
])
def test_split_token_budget_preserves_total(total, expected):
    assert split_token_budget(total, 0.60) == expected
    assert sum(expected) == total


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True])
def test_split_token_budget_rejects_invalid_totals(invalid):
    with pytest.raises(ValueError):
        split_token_budget(invalid, 0.60)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_split_token_budget_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError):
        split_token_budget(10, fraction)


def test_run_measured_calls_all_succeed(tmp_path):
    cache_path = tmp_path / "crash.json"
    calls = []

    def call(run_i):
        calls.append(run_i)
        return run_i * 2

    samples, status, partial_text, metadata = Shared.run_measured_calls(
        3, call, "tag", {}, cache_path, "testing", _engine())
    assert samples == [0, 2, 4]
    assert status == "ok"
    assert partial_text == ""
    assert metadata == {"budget_nudged": False}
    assert calls == [0, 1, 2]


def test_run_measured_calls_observes_pause_before_every_attempt(tmp_path, monkeypatch):
    pauses = []
    monkeypatch.setattr(shared, "wait_if_paused", lambda: pauses.append("boundary"))
    Shared.run_measured_calls(
        3, lambda run_i: run_i, "tag", {}, tmp_path / "crash.json", "testing", _engine(),
    )
    assert pauses == ["boundary", "boundary", "boundary"]


def _measurement(implausible=False):
    return GenerationMeasurement(
        client_ttft_sec=0.1, generated_tokens=10, tokens_per_sec=20,
        client_wall_sec=0.6, decode_sec=0.5,
        server_tps_implausible=implausible,
    )


def test_retry_implausible_tps_retries_once_and_uses_valid_retry(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    outcomes = iter([_measurement(True), _measurement(False)])
    result = Shared.retry_implausible_tps(lambda: next(outcomes), "model", "llm")
    assert result.server_tps_implausible is False
    output = capsys.readouterr().out
    assert '"status":"retrying"' in output
    assert '"status":"valid"' in output


def test_retry_implausible_tps_returns_second_invalid_measurement_without_third_call(monkeypatch, capsys):
    monkeypatch.setenv("LOCAL_AI_BENCH_PROGRESS", "1")
    calls = []

    def call():
        calls.append(True)
        return _measurement(True)

    result = Shared.retry_implausible_tps(call, "model", "llm")
    assert result.server_tps_implausible is True
    assert len(calls) == 2
    output = capsys.readouterr().out
    assert '"status":"retrying"' in output
    assert '"status":"invalid"' in output


def test_run_measured_calls_timeout_stops_immediately(tmp_path):
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        if run_i == 1:
            raise TimeoutError("timed out")
        return run_i

    samples, status, partial_text, metadata = Shared.run_measured_calls(
        3, call, "tag", {}, cache_path, "testing", _engine())
    assert samples == [0]
    assert status == "timed_out"
    assert partial_text == ""
    assert metadata == {"budget_nudged": False}


def test_run_measured_calls_timeout_captures_partial_text(tmp_path):
    """A timeout must surface the partial text, not just a bare 'timed_out' status."""
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        raise EngineTimeout("timed out", partial_text="The answer is B")

    samples, status, partial_text, metadata = Shared.run_measured_calls(
        3, call, "tag", {}, cache_path, "testing", _engine())
    assert samples == []
    assert status == "timed_out"
    assert partial_text == "The answer is B"
    assert metadata == {"budget_nudged": False}


def test_run_measured_calls_loop_detected_is_a_distinct_status(tmp_path):
    """EngineLoopDetected must surface as "loop_detected", not fold into "timed_out"."""
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        raise EngineLoopDetected("detected a generation loop after 8s", partial_text="wait, wait, wait,")

    samples, status, partial_text, metadata = Shared.run_measured_calls(
        3, call, "tag", {}, cache_path, "testing", _engine())
    assert samples == []
    assert status == "loop_detected"
    assert partial_text == "wait, wait, wait,"
    assert metadata == {"budget_nudged": False}


def test_run_measured_calls_budget_exhaustion_is_distinct_and_preserves_metadata(tmp_path):
    def call(_run_i):
        raise EngineBudgetExceeded("budget exhausted", partial_text="Answer: C")

    samples, status, partial_text, metadata = Shared.run_measured_calls(
        1, call, "tag", {}, tmp_path / "crash.json", "testing", _engine())
    assert samples == []
    assert status == "budget_exceeded"
    assert partial_text == "Answer: C"
    assert metadata == {"budget_nudged": True}


def test_run_measured_calls_ordinary_failure_skips_and_continues(tmp_path):
    cache_path = tmp_path / "crash.json"

    def call(run_i):
        if run_i == 1:
            raise ValueError("some ordinary failure")
        return run_i

    samples, status, _, _metadata = Shared.run_measured_calls(
        3, call, "tag", {}, cache_path, "testing", _engine())
    # run_i=1 fails but still counts as attempted (advances), so only 2 samples collected
    assert samples == [0, 2]
    assert status == "ok"


def test_run_measured_calls_crash_retries_then_gives_up(tmp_path):
    cache_path = tmp_path / "crash.json"
    crash_cache = {}

    # Recovery always succeeds so the retry loop doesn't stall — the crash is
    # deterministic, so it exhausts CRASH_RETRY_MAX and gives up regardless.
    def call(run_i):
        raise requests.exceptions.ConnectionError("actively refused")

    samples, status, _, _metadata = Shared.run_measured_calls(
        3, call, "tag", crash_cache, cache_path, "testing", _engine())
    assert samples == []
    assert status == "crashed"
    assert "tag" in crash_cache["fake"]
    assert (Shared.load_crash_cache(cache_path)["fake"]["tag"]["crashed_at"]
            == crash_cache["fake"]["tag"]["crashed_at"])


def test_run_measured_calls_crash_recovers_and_retries_same_run(tmp_path):
    cache_path = tmp_path / "crash.json"

    attempts = {"n": 0}

    def call(run_i):
        if run_i == 0 and attempts["n"] == 0:
            attempts["n"] += 1
            raise requests.exceptions.ConnectionError("actively refused")
        return run_i

    samples, status, _, _metadata = Shared.run_measured_calls(
        2, call, "tag", {}, cache_path, "testing", _engine())
    assert samples == [0, 1]
    assert status == "ok"


def test_run_measured_calls_crash_gives_up_if_recovery_fails(tmp_path):
    cache_path = tmp_path / "crash.json"
    crash_cache = {}

    def call(run_i):
        raise requests.exceptions.ConnectionError("actively refused")

    samples, status, _, _metadata = Shared.run_measured_calls(
        3, call, "tag", crash_cache, cache_path, "testing", _engine(recovers=False))
    assert samples == []
    assert status == "crashed"
    assert "tag" in crash_cache["fake"]


def test_slow_tps_early_exit_triggers_below_cutoff():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "2K", True, [config.SLOW_MODEL_MIN_TPS - 1.0], False,
    )
    assert stopped is True
    assert results["short"]["slow_tps"] == "2K"


def test_slow_tps_early_exit_force_all_ignores_cutoff():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "2K", True, [config.SLOW_MODEL_MIN_TPS - 1.0], True,
    )
    assert stopped is False
    assert "slow_tps" not in results["short"]


def test_slow_tps_early_exit_not_first_ctx_never_triggers():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "8K", False, [config.SLOW_MODEL_MIN_TPS - 1.0], False,
    )
    assert stopped is False
    assert "slow_tps" not in results["short"]


def test_slow_tps_early_exit_above_cutoff_does_not_trigger():
    results = {"short": {}}
    stopped = Shared.slow_tps_early_exit(
        results, "short", "Model", "2K", True, [config.SLOW_MODEL_MIN_TPS + 10.0], False,
    )
    assert stopped is False
    assert "slow_tps" not in results["short"]
