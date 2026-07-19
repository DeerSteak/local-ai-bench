import config
from concurrency_benchmark import ConcurrencyBenchmark


def test_below_floor_never_stops_even_if_slow():
    # 8-way concurrency (config.CONCURRENCY_MIN_LEVEL_BEFORE_SOFT_EXIT) is a
    # common agentic/tool-calling fan-out shape, worth a real data point even
    # if a lower level already looked slow.
    assert ConcurrencyBenchmark.should_stop_escalating(
        1, mean_tps=1.0, force_all=False) is False
    assert ConcurrencyBenchmark.should_stop_escalating(
        4, mean_tps=1.0, force_all=False) is False


def test_at_floor_stops_when_below_cutoff():
    level = config.CONCURRENCY_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        level, mean_tps=config.SLOW_MODEL_MIN_TPS - 1, force_all=False) is True


def test_at_floor_does_not_stop_when_above_cutoff():
    level = config.CONCURRENCY_MIN_LEVEL_BEFORE_SOFT_EXIT
    assert ConcurrencyBenchmark.should_stop_escalating(
        level, mean_tps=config.SLOW_MODEL_MIN_TPS + 1, force_all=False) is False


def test_above_floor_stops_when_below_cutoff():
    assert ConcurrencyBenchmark.should_stop_escalating(
        64, mean_tps=config.SLOW_MODEL_MIN_TPS - 1, force_all=False) is True


def test_force_all_disables_soft_exit_even_past_floor():
    assert ConcurrencyBenchmark.should_stop_escalating(
        64, mean_tps=0.1, force_all=True) is False
