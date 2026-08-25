import pytest

from scripts.workloads.sustained_benchmark import (
    aligned_sustained_windows, sustained_measurement_valid,
)


def blocks():
    memory = {"windows": [{
        "name": "measured:sustained", "samples": [
            {"timestamp_sec": 5.0, "host_ram_used_gb": 20, "process_rss_gb": 4},
            {"timestamp_sec": 9.9, "host_ram_used_gb": 22, "process_rss_gb": 6},
            {"timestamp_sec": 15.1, "host_ram_used_gb": 24, "process_rss_gb": 8},
        ],
    }]}
    power = {"windows": [{
        "name": "measured:sustained", "samples": [
            {"timestamp_sec": 5.0, "watts": 100},
            {"timestamp_sec": 9.9, "watts": 120},
            {"timestamp_sec": 15.1, "watts": 90},
        ],
    }]}
    temperature = {"windows": [{
        "name": "measured:sustained", "samples": [
            {"timestamp_sec": 5.0, "soc_package_c": 60, "gpu_die_c": 60},
            {"timestamp_sec": 9.9, "soc_package_c": 64, "gpu_die_c": 64},
            {"timestamp_sec": 15.1, "soc_package_c": 70, "gpu_die_c": 70},
        ],
    }]}
    return memory, power, temperature


def test_aligned_windows_distribute_requests_by_real_overlap_and_share_telemetry_boundaries():
    requests = [
        {"start_sec": 0, "end_sec": 5, "generated_tokens": 50},
        {"start_sec": 5, "end_sec": 15, "generated_tokens": 200},
        {"start_sec": 16, "end_sec": 20, "generated_tokens": 40},
    ]
    memory, power, temperature = blocks()
    windows = aligned_sustained_windows(
        requests, 20, 10, measured_offset_sec=5,
        memory=memory, power=power, temperature=temperature,
    )
    assert len(windows) == 2
    assert windows[0]["tokens"] == 150
    assert windows[0]["tokens_per_sec"] == 15
    assert windows[1]["tokens"] == 140
    assert windows[1]["tokens_per_sec"] == 14
    assert windows[0]["host_ram_used_gb"] == 21
    assert windows[1]["host_ram_used_gb"] == 24
    assert windows[0]["power_watts"] == 110
    assert windows[1]["power_watts"] == 90
    assert windows[0]["soc_package_c"] == 62
    assert windows[1]["soc_package_c"] == 70
    assert windows[0]["gpu_die_c"] == 62
    assert windows[1]["gpu_die_c"] == 70


def test_trailing_partial_window_uses_its_actual_duration():
    windows = aligned_sustained_windows(
        [{"start_sec": 0, "end_sec": 12, "generated_tokens": 120}], 12, 10,
    )
    assert [(window["timestamp_sec"], window["duration_sec"], window["tokens_per_sec"])
            for window in windows] == [(0, 10, 10), (10, 2, 10)]


def test_gaps_are_zero_throughput_instead_of_disappearing():
    windows = aligned_sustained_windows(
        [{"start_sec": 0, "end_sec": 5, "generated_tokens": 50}], 20, 10,
    )
    assert [window["tokens_per_sec"] for window in windows] == [5, 0]


def test_invalid_requests_are_ignored_without_poisoning_valid_intervals():
    requests = [
        {"start_sec": 0, "end_sec": 10, "generated_tokens": 100},
        {"start_sec": 10, "end_sec": 10, "generated_tokens": 100},
        {"start_sec": 10, "end_sec": 20, "generated_tokens": -1},
        {"start_sec": "private", "end_sec": 20, "generated_tokens": 10},
    ]
    assert [window["tokens_per_sec"] for window in aligned_sustained_windows(
        requests, 20, 10,
    )] == [10, 0]


def test_missing_temperature_or_power_stays_unknown_on_the_shared_axis():
    windows = aligned_sustained_windows([], 10, 10)
    assert windows[0]["tokens_per_sec"] == 0
    assert windows[0]["power_watts"] is None
    assert windows[0]["soc_package_c"] is None
    assert windows[0]["cpu_package_c"] is None
    assert windows[0]["gpu_die_c"] is None
    assert windows[0]["gpu_hotspot_c"] is None


@pytest.mark.parametrize(("requests", "valid", "expected"), [
    (0, 0, False),
    (10, 0, False),
    (10, 9, False),
    (10, 10, True),
])
def test_measurement_requires_a_nonempty_fully_valid_request_series(requests, valid, expected):
    assert sustained_measurement_valid(requests, valid) is expected


@pytest.mark.parametrize(("duration", "window"), [(0, 10), (-1, 10), (10, 0), (10, -1)])
def test_invalid_window_configuration_is_rejected(duration, window):
    with pytest.raises(ValueError, match="positive"):
        aligned_sustained_windows([], duration, window)
