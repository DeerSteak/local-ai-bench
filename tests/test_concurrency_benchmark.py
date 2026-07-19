import config
from concurrency_benchmark import ConcurrencyBenchmark


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
