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
