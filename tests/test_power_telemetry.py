from pathlib import Path

import pytest

from scripts.runtime.telemetry import (
    PowerReading, efficiency_per_joule, integrate_power_joules,
    parse_nvidia_power, parse_powermetrics_power, parse_rapl_energy_uj,
    parse_rocm_power,
)


FIXTURES = Path(__file__).parent / "fixtures" / "power"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_powermetrics_prefers_combined_package_power_and_normalizes_milliwatts():
    assert parse_powermetrics_power(fixture("powermetrics.txt")) == PowerReading(
        1.713, "powermetrics", "processor_package",
    )
    assert parse_powermetrics_power(
        "Package Power: 12.5 W\nCombined Power (CPU + GPU + ANE): 3.25 W\n"
    ) == PowerReading(3.25, "powermetrics", "processor_package")


def test_powermetrics_rejects_partial_and_permission_error_output():
    assert parse_powermetrics_power("CPU Power: 2 W\nGPU Power: 1 W") is None
    assert parse_powermetrics_power("powermetrics: permission denied") is None
    assert parse_powermetrics_power("Combined Power (CPU + GPU + ANE): nan W") is None


def test_nvidia_power_sums_devices_and_rejects_non_measurements():
    assert parse_nvidia_power(fixture("nvidia-smi.txt")) == PowerReading(
        355.75, "nvidia-smi", "accelerator",
    )
    assert parse_nvidia_power("245.5 W\nN/A\n[Not Supported]\n") == PowerReading(
        245.5, "nvidia-smi", "accelerator",
    )
    assert parse_nvidia_power("N/A\nGPU is lost\n") is None
    assert parse_nvidia_power("-1\ninf\n") is None


def test_rocm_power_sums_devices_and_rejects_malformed_or_missing_values():
    assert parse_rocm_power(fixture("rocm-smi.json")) == PowerReading(
        279.75, "rocm-smi", "accelerator",
    )
    assert parse_rocm_power('{"card0": {"Temperature": 40}}') is None
    assert parse_rocm_power('{"card0":') is None
    assert parse_rocm_power('[{"Average Graphics Package Power (W)": 4}]') is None


def test_rapl_parser_converts_microjoules_and_rejects_invalid_counters():
    assert parse_rapl_energy_uj(fixture("rapl-energy-uj.txt")) == pytest.approx(123.456789)
    assert parse_rapl_energy_uj("permission denied") is None
    assert parse_rapl_energy_uj("-1") is None


def test_trapezoidal_integration_uses_actual_uneven_timestamps():
    assert integrate_power_joules([(0, 10), (0.25, 14), (1.0, 18)]) == pytest.approx(15.0)


def test_integration_does_not_bridge_unknown_or_invalid_intervals():
    assert integrate_power_joules([(0, 10), (1, None), (2, 20), (3, 30)]) == 25
    assert integrate_power_joules([(0, 10)]) is None
    assert integrate_power_joules([]) is None
    assert integrate_power_joules([(1, 10), (1, 20)]) is None
    assert integrate_power_joules([(0, float("nan")), (1, 20)]) is None


@pytest.mark.parametrize(
    ("work", "energy", "expected"),
    [(100, 20, 5), (0, 20, None), (100, 0, None), (None, 20, None),
     (100, None, None), (-1, 20, None), (100, float("nan"), None)],
)
def test_efficiency_per_joule_guards_unknown_zero_and_invalid_inputs(work, energy, expected):
    assert efficiency_per_joule(work, energy) == expected


def test_efficiency_rejects_boolean_values():
    assert efficiency_per_joule(True, 10) is None
