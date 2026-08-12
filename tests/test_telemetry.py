import time

import pytest

from scripts.runtime import config
from scripts.runtime.telemetry import (
    TelemetrySample, TelemetrySampler, calculate_headroom,
    summarize_case, summarize_samples, summarize_windows,
)


def sample(timestamp, window="measured", host=None, rss=None, used=None, total=None):
    return TelemetrySample(timestamp, window, host, rss, used, total)


def test_empty_and_all_failed_aggregation_remain_unknown():
    empty = summarize_samples("measured", [])
    failed = summarize_samples("measured", [sample(0), sample(1)])
    assert empty.sample_count == 0
    assert empty.duration_sec == 0
    assert failed.sample_count == 2
    for summary in failed.channels.values():
        assert summary.peak_gb is None
        assert summary.mean_gb is None
        assert summary.final_gb is None
        assert summary.valid_samples == 0


def test_single_sample_summary_has_zero_duration():
    summary = summarize_samples("idle", [sample(2.0, "idle", host=10, rss=2)])
    assert summary.duration_sec == 0
    assert summary.channels["host_ram_used_gb"].peak_gb == 10
    assert summary.channels["process_rss_gb"].mean_gb == 2


def test_many_windows_are_retained_and_case_summary_is_weighted():
    samples = [
        sample(0, "idle", host=10, rss=1),
        sample(1, "idle", host=12, rss=2),
        sample(2, "model_load", host=20, rss=6),
        sample(3, "measured:prefill", host=16, rss=4),
        sample(5, "measured:decode", host=18, rss=5),
    ]
    windows = summarize_windows(samples)
    summary = summarize_case(windows)
    assert [window.name for window in windows] == [
        "idle", "model_load", "measured:prefill", "measured:decode",
    ]
    assert [window.sample_count for window in windows] == [2, 1, 1, 1]
    assert summary["host_ram_used_gb"].peak_gb == 20
    assert summary["host_ram_used_gb"].mean_gb == pytest.approx(15.2)
    assert summary["process_rss_gb"].final_gb == 5


@pytest.mark.parametrize(("peak", "ceiling", "absolute", "fraction", "state"), [
    (None, 10, None, None, "unknown"),
    (5, None, None, None, "unknown"),
    (5, 0, None, None, "unknown"),
    (8, 10, 2, 0.2, "comfortable"),
    (8.01, 10, 1.99, 0.199, "tight"),
    (10, 10, 0, 0, "tight"),
    (10.01, 10, -0.01, -0.001, "exceeded"),
])
def test_headroom_boundaries(peak, ceiling, absolute, fraction, state):
    assert config.MEMORY_HEADROOM_COMFORTABLE_FRACTION == 0.20
    result = calculate_headroom(peak, ceiling)
    assert result.state == state
    assert result.absolute_gb == (None if absolute is None else pytest.approx(absolute))
    assert result.fraction == (None if fraction is None else pytest.approx(fraction))


def test_sampler_cleans_up_after_context_exception():
    sampler = TelemetrySampler(
        42, interval_sec=0.001,
        sample_fn=lambda timestamp, window: sample(timestamp, window, host=10),
    )
    with pytest.raises(RuntimeError):
        with sampler:
            time.sleep(0.005)
            raise RuntimeError("case failed")
    assert sampler._thread is not None
    assert not sampler._thread.is_alive()
    assert sampler.samples


def test_sampler_failure_records_unknown_and_continues():
    calls = 0

    def raising_then_valid(timestamp, window):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("sensor unavailable")
        return sample(timestamp, window, host=12)

    sampler = TelemetrySampler(42, interval_sec=0.001, sample_fn=raising_then_valid).start()
    time.sleep(0.006)
    samples = sampler.stop()
    assert sampler.failed_samples == 1
    assert samples[0].host_ram_used_gb is None
    assert any(item.host_ram_used_gb == 12 for item in samples[1:])


def test_sampler_switches_windows_without_losing_samples():
    sampler = TelemetrySampler(
        42, interval_sec=0.001,
        sample_fn=lambda timestamp, window: sample(timestamp, window, rss=2),
    ).start()
    time.sleep(0.003)
    sampler.set_window("model_load")
    time.sleep(0.003)
    sampler.set_window("measured")
    time.sleep(0.003)
    windows = {item.window for item in sampler.stop()}
    assert windows == {"idle", "model_load", "measured"}


@pytest.mark.parametrize("interval", [0, -0.1])
def test_sampler_rejects_nonpositive_intervals(interval):
    with pytest.raises(ValueError, match="positive"):
        TelemetrySampler(42, interval_sec=interval)
