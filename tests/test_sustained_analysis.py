import pytest

from scripts.runtime import config
from scripts.workloads.sustained_analysis import (
    analyze_sustained_series, correlate_cause, performance_classification, throttle_onset,
)


def series(rates, *, step=20, temperatures=None, powers=None):
    temperatures = temperatures or [None] * len(rates)
    powers = powers or [None] * len(rates)
    return [
        {"timestamp_sec": index * step, "duration_sec": step, "tokens_per_sec": rate,
         "gpu_die_c": temperatures[index], "power_watts": powers[index]}
        for index, rate in enumerate(rates)
    ]


def analyze(windows, **kwargs):
    return analyze_sustained_series(
        windows, minimum_duration_sec=120, initial_count=3, late_count=3, **kwargs,
    )


def test_flat_series_is_stable_with_full_retention_and_no_onset():
    result = analyze(series([100] * 9, temperatures=[60] * 9))
    assert result["initial_tokens_per_sec"] == 100
    assert result["steady_state_tokens_per_sec"] == 100
    assert result["retention_ratio"] == 1
    assert result["throttle_onset_sec"] is None
    assert result["performance"] == "stable"
    assert result["cause"] == "neither"


def test_monotonic_decline_reports_retention_and_first_sustained_onset():
    result = analyze(series([100, 100, 100, 94, 92, 90, 82, 80, 78]))
    assert result["initial_tokens_per_sec"] == 100
    assert result["steady_state_tokens_per_sec"] == 80
    assert result["retention_ratio"] == 0.8
    assert result["throttle_onset_sec"] == 60
    assert result["performance"] == "significant_degradation"
    assert result["cause"] == "unavailable"
    assert result["ordinal_drift"] == "declining"


def test_single_dip_then_recovery_does_not_trigger_sustained_onset():
    windows = series([100, 100, 100, 80, 100, 100, 100, 100, 100])
    assert throttle_onset(windows, 100, tolerance_fraction=0.05, consecutive=3) is None
    result = analyze(windows)
    assert result["performance"] == "stable"
    assert result["throttle_onset_sec"] is None
    assert result["ordinal_drift"] == "none"


def test_noisy_but_stable_series_does_not_cross_retention_or_onset_thresholds():
    result = analyze(series([100, 98, 102, 96, 101, 97, 96, 98, 97]))
    assert result["retention_ratio"] == pytest.approx(0.97)
    assert result["performance"] == "stable"
    assert result["throttle_onset_sec"] is None


@pytest.mark.parametrize("windows", [
    series([100] * 5, step=30),
    series([100] * 9, step=10),
    [{"timestamp_sec": index * 20, "tokens_per_sec": None} for index in range(9)],
    series([100] * 8 + [-1]),
    [{"timestamp_sec": 0, "tokens_per_sec": 100}] * 9,
])
def test_short_invalid_or_non_increasing_series_is_indeterminate(windows):
    result = analyze(windows)
    assert result["performance"] == "indeterminate"
    assert result["retention_ratio"] is None
    assert result["cause"] == "unavailable"


@pytest.mark.parametrize(("ratio", "expected"), [
    (None, "indeterminate"),
    (config.SUSTAINED_SIGNIFICANT_RETENTION - 0.001, "significant_degradation"),
    (config.SUSTAINED_SIGNIFICANT_RETENTION, "mild_degradation"),
    (config.SUSTAINED_MILD_RETENTION - 0.001, "mild_degradation"),
    (config.SUSTAINED_MILD_RETENTION, "stable"),
    (1.05, "stable"),
])
def test_performance_classification_boundaries(ratio, expected):
    assert performance_classification(ratio) == expected


def degraded(temperatures=None, powers=None):
    return series(
        [100, 100, 100, 80, 80, 80, 80, 80, 80],
        temperatures=temperatures, powers=powers,
    )


@pytest.mark.parametrize(("temperatures", "powers", "expected"), [
    ([50, 51, 52, 70, 70, 70, 70, 70, 70], [100, 100, 100, 80, 80, 80, 80, 80, 80],
     "temperature_and_power_correlated"),
    ([50, 51, 52, 70, 70, 70, 70, 70, 70], [100] * 9, "temperature_correlated"),
    ([60] * 9, [100, 100, 100, 80, 80, 80, 80, 80, 80], "power_correlated"),
    ([60] * 9, [100] * 9, "neither"),
    (None, None, "unavailable"),
])
def test_cause_classification_keeps_temperature_power_neither_and_unavailable_distinct(
        temperatures, powers, expected):
    assert analyze(degraded(temperatures, powers))["cause"] == expected


def test_hot_stable_device_is_not_called_temperature_correlated():
    windows = series([100] * 9, temperatures=[90] * 9, powers=[120] * 9)
    assert correlate_cause(
        windows, "stable", None, initial_count=3, late_count=3,
    ) == "neither"


def test_temperature_rise_without_a_ceiling_is_not_called_thermal():
    temperatures = [50, 51, 52, 60, 63, 66, 68, 71, 75]
    assert analyze(degraded(temperatures, [100] * 9))["cause"] == "neither"


def test_power_decline_later_than_throughput_onset_is_not_called_correlated():
    powers = [100, 100, 100, 100, 100, 100, 80, 80, 80]
    assert analyze(degraded([60] * 9, powers))["cause"] == "neither"


def test_trial_drift_reference_is_preserved_without_changing_soak_classification():
    result = analyze(series([100] * 9), related_trial_drift="declining")
    assert result["performance"] == "stable"
    assert result["related_trial_drift"] == "declining"
