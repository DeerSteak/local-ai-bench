"""Pure sustained-throughput degradation and telemetry correlation analysis."""

import math
from typing import Mapping, Sequence

from scripts.runtime import config
from scripts.results.trial_set import monotonic_drift


TEMPERATURE_KEYS = ("soc_package_c", "cpu_package_c", "gpu_die_c", "gpu_hotspot_c")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def performance_classification(retention_ratio: float | None) -> str:
    if retention_ratio is None:
        return "indeterminate"
    if retention_ratio < config.SUSTAINED_SIGNIFICANT_RETENTION:
        return "significant_degradation"
    if retention_ratio < config.SUSTAINED_MILD_RETENTION:
        return "mild_degradation"
    return "stable"


def throttle_onset(windows: Sequence[Mapping[str, object]], initial_throughput: float,
                   *, tolerance_fraction: float = config.SUSTAINED_ONSET_TOLERANCE_FRACTION,
                   consecutive: int = config.SUSTAINED_ONSET_CONSECUTIVE_WINDOWS,
                   baseline_count: int = config.SUSTAINED_INITIAL_WINDOWS) \
        -> float | None:
    if (initial_throughput <= 0 or consecutive <= 0 or baseline_count < 0
            or not 0 <= tolerance_fraction < 1):
        return None
    threshold = initial_throughput * (1 - tolerance_fraction)
    for index in range(baseline_count, len(windows) - consecutive + 1):
        run = windows[index:index + consecutive]
        values = [_number(window.get("tokens_per_sec")) for window in run]
        if all(value is not None and value < threshold for value in values):
            return _number(run[0].get("timestamp_sec"))
    return None


def _channel_values(windows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    return [value for window in windows if (value := _number(window.get(key))) is not None]


def _temperature_correlated(windows: Sequence[Mapping[str, object]], initial_count: int,
                            late_count: int, onset: float | None) -> bool:
    if onset is None:
        return False
    for key in TEMPERATURE_KEYS:
        early = _channel_values(windows[:initial_count], key)
        late = _channel_values(windows[-late_count:], key)
        all_values = _channel_values(windows, key)
        if not early or len(late) < 2 or not all_values:
            continue
        late_mean = _mean(late)
        early_mean = _mean(early)
        onset_values = [
            value for window in windows
            if (_number(window.get("timestamp_sec")) or 0) >= onset
            and (value := _number(window.get(key))) is not None
        ]
        if (late_mean is not None and early_mean is not None
                and late_mean - early_mean >= config.SUSTAINED_TEMPERATURE_RISE_C
                and max(late) - min(late) <= config.SUSTAINED_TEMPERATURE_CEILING_BAND_C
                and onset_values
                and onset_values[0] >= max(all_values) - config.SUSTAINED_TEMPERATURE_CEILING_BAND_C):
            return True
    return False


def _power_correlated(windows: Sequence[Mapping[str, object]], initial_count: int,
                      late_count: int, onset: float | None) -> bool:
    if onset is None:
        return False
    early = _channel_values(windows[:initial_count], "power_watts")
    late = _channel_values(windows[-late_count:], "power_watts")
    early_mean = _mean(early)
    late_mean = _mean(late)
    threshold = early_mean * (1 - config.SUSTAINED_POWER_DROP_FRACTION) if early_mean else None
    onset_values = [
        value for window in windows
        if (_number(window.get("timestamp_sec")) or 0) >= onset
        and (value := _number(window.get("power_watts"))) is not None
    ]
    return bool(threshold is not None and late_mean is not None and onset_values
                and late_mean <= threshold and onset_values[0] <= threshold)


def correlate_cause(windows: Sequence[Mapping[str, object]], performance: str,
                    onset: float | None, *, initial_count: int, late_count: int) -> str:
    temperature_available = any(_channel_values(windows, key) for key in TEMPERATURE_KEYS)
    power_available = bool(_channel_values(windows, "power_watts"))
    if not temperature_available and not power_available:
        return "unavailable"
    if performance not in {"mild_degradation", "significant_degradation"}:
        return "neither"
    temperature = _temperature_correlated(windows, initial_count, late_count, onset)
    power = _power_correlated(windows, initial_count, late_count, onset)
    if temperature and power:
        return "temperature_and_power_correlated"
    if temperature:
        return "temperature_correlated"
    if power:
        return "power_correlated"
    return "neither"


def analyze_sustained_series(
        windows: Sequence[Mapping[str, object]], *,
        minimum_duration_sec: float = config.SUSTAINED_MIN_CLASSIFICATION_SEC,
        initial_count: int = config.SUSTAINED_INITIAL_WINDOWS,
        late_count: int = config.SUSTAINED_STEADY_WINDOWS,
        measurement_valid: bool = True,
        related_trial_drift: str | None = None) -> dict[str, object]:
    throughput = [_number(window.get("tokens_per_sec")) for window in windows]
    timestamps = [_number(window.get("timestamp_sec")) for window in windows]
    valid_throughput = [value for value in throughput if value is not None and value >= 0]
    valid_timestamps = [value for value in timestamps if value is not None]
    valid_timeline = (
        len(windows) >= initial_count + late_count
        and len(valid_throughput) == len(windows)
        and len(valid_timestamps) == len(windows)
        and all(after > before
                for before, after in zip(valid_timestamps, valid_timestamps[1:]))
    )
    last_window_duration = _number(windows[-1].get("duration_sec")) if windows else None
    duration = (
        valid_timestamps[-1] + max(last_window_duration or 0, 0) - valid_timestamps[0]
        if valid_timeline else None
    )
    sufficiently_long = bool(measurement_valid and valid_timeline and duration is not None
                             and duration >= minimum_duration_sec)
    ordinal_drift = monotonic_drift(valid_throughput) \
        if len(valid_throughput) == len(windows) else "insufficient"
    if not sufficiently_long:
        return {
            "initial_tokens_per_sec": None,
            "steady_state_tokens_per_sec": None,
            "retention_ratio": None,
            "throttle_onset_sec": None,
            "performance": "indeterminate",
            "cause": "unavailable",
            "duration_sec": duration,
            "window_count": len(windows),
            "ordinal_drift": ordinal_drift,
            "related_trial_drift": related_trial_drift,
        }
    initial = _mean(valid_throughput[:initial_count])
    steady = _mean(valid_throughput[-late_count:])
    retention = steady / initial if initial and steady is not None else None
    performance = performance_classification(retention)
    onset = throttle_onset(windows, initial, baseline_count=initial_count) \
        if initial is not None else None
    return {
        "initial_tokens_per_sec": initial,
        "steady_state_tokens_per_sec": steady,
        "retention_ratio": retention,
        "throttle_onset_sec": onset,
        "performance": performance,
        "cause": correlate_cause(
            windows, performance, onset, initial_count=initial_count, late_count=late_count,
        ),
        "duration_sec": duration,
        "window_count": len(windows),
        "ordinal_drift": ordinal_drift,
        "related_trial_drift": related_trial_drift,
    }
