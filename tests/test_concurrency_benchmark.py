import config
from concurrency_benchmark import ConcurrencyBenchmark
from shared import Shared


def test_below_floor_never_stops_even_if_slow():
    floor = config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        1, mean_tps=1.0, force_all=False, soft_exit_floor=floor) is False
    assert ConcurrencyBenchmark.should_stop_escalating(
        floor - 1, mean_tps=1.0, force_all=False, soft_exit_floor=floor) is False


def test_at_floor_stops_when_below_cutoff():
    floor = config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        floor, mean_tps=config.SLOW_MODEL_MIN_TPS - 1, force_all=False, soft_exit_floor=floor) is True


def test_at_floor_does_not_stop_when_above_cutoff():
    floor = config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        floor, mean_tps=config.SLOW_MODEL_MIN_TPS + 1, force_all=False, soft_exit_floor=floor) is False


def test_above_floor_stops_when_below_cutoff():
    floor = config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        32, mean_tps=config.SLOW_MODEL_MIN_TPS - 1, force_all=False, soft_exit_floor=floor) is True


def test_force_all_disables_soft_exit_even_past_floor():
    floor = config.CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        32, mean_tps=0.1, force_all=True, soft_exit_floor=floor) is False


def test_none_floor_never_stops_regardless_of_level_or_speed():
    # The tool test's shape: soft_exit_floor=None means only a hard stop
    # (crash/load-failure) ends the sweep early, never the tok/s cutoff.
    assert ConcurrencyBenchmark.should_stop_escalating(
        8, mean_tps=0.01, force_all=False, soft_exit_floor=None) is False
    assert ConcurrencyBenchmark.should_stop_escalating(
        1, mean_tps=0.01, force_all=False, soft_exit_floor=None) is False


class _FakeEngine:
    name = "fake"

    def __init__(self):
        self.seen_prompts = []

    def generate(self, tag, prompt, timeout, per_request_context, level):
        self.seen_prompts.append(prompt)
        return (0.1, 10, 100.0)


def test_fire_batch_returns_one_sample_per_concurrent_request():
    engine = _FakeEngine()
    samples = ConcurrencyBenchmark._fire_batch(engine, "tag", 4, 2048)
    assert samples == [(0.1, 10, 100.0)] * 4


def test_fire_batch_gives_each_concurrent_request_a_distinct_prompt():
    # Without a unique nonce per request, an engine's prefix cache would
    # serve some requests near-instantly regardless of real concurrency —
    # see Shared.build_prompt_for_context.
    engine = _FakeEngine()
    ConcurrencyBenchmark._fire_batch(engine, "tag", 5, 512)
    assert len(engine.seen_prompts) == 5
    assert len(set(engine.seen_prompts)) == 5


def test_fire_batch_propagates_a_request_failure():
    class CrashingEngine:
        name = "fake"

        def generate(self, tag, prompt, timeout, per_request_context, level):
            raise RuntimeError("boom")

    try:
        ConcurrencyBenchmark._fire_batch(CrashingEngine(), "tag", 2, 512)
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as e:
        assert "boom" in str(e)


class _RetryEngine:
    def __init__(self, recovers=True):
        self.recovers = recovers
        self.recovery_calls = 0

    def is_connection_crash(self, error):
        return isinstance(error, ConnectionError)

    def wait_for_recovery(self):
        self.recovery_calls += 1
        return self.recovers


def test_fire_batch_retries_transient_connection_crash(monkeypatch):
    outcomes = iter([ConnectionError("down"), [(0.1, 2, 3.0)]])

    def fire(*args):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ConcurrencyBenchmark, "_fire_batch", staticmethod(fire))
    engine = _RetryEngine()
    samples, status, error, elapsed = ConcurrencyBenchmark._fire_batch_with_crash_retries(
        engine, "tag", 1, 512,
    )
    assert samples == [(0.1, 2, 3.0)]
    assert status == "ok"
    assert error is None
    assert elapsed >= 0
    assert engine.recovery_calls == 1


def test_fire_batch_reports_crash_only_after_retry_limit(monkeypatch):
    calls = []

    def fire(*args):
        calls.append(True)
        raise ConnectionError("down")

    monkeypatch.setattr(ConcurrencyBenchmark, "_fire_batch", staticmethod(fire))
    engine = _RetryEngine()
    samples, status, error, elapsed = ConcurrencyBenchmark._fire_batch_with_crash_retries(
        engine, "tag", 2, 512,
    )
    assert samples == []
    assert status == "crashed"
    assert isinstance(error, ConnectionError)
    assert elapsed == 0
    assert len(calls) == Shared.CRASH_RETRY_MAX + 1


def test_fire_batch_does_not_retry_non_connection_failure(monkeypatch):
    monkeypatch.setattr(
        ConcurrencyBenchmark, "_fire_batch",
        staticmethod(lambda *args: (_ for _ in ()).throw(ValueError("bad response"))),
    )
    samples, status, error, elapsed = ConcurrencyBenchmark._fire_batch_with_crash_retries(
        _RetryEngine(), "tag", 2, 512,
    )
    assert samples == []
    assert status == "failed"
    assert isinstance(error, ValueError)
    assert elapsed == 0


def test_fire_batch_stops_when_engine_cannot_recover(monkeypatch):
    monkeypatch.setattr(
        ConcurrencyBenchmark, "_fire_batch",
        staticmethod(lambda *args: (_ for _ in ()).throw(ConnectionError("down"))),
    )
    engine = _RetryEngine(recovers=False)
    _, status, error, _ = ConcurrencyBenchmark._fire_batch_with_crash_retries(
        engine, "tag", 2, 512,
    )
    assert status == "crashed"
    assert isinstance(error, ConnectionError)
    assert engine.recovery_calls == 1
