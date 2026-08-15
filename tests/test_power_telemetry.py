from pathlib import Path

import pytest

from scripts.runtime.telemetry import (
    PowerAvailability, PowerReading, discover_power_source, efficiency_per_joule,
    integrate_power_joules,
    parse_nvidia_power, parse_powermetrics_power, parse_rapl_energy_uj,
    parse_rocm_power, query_power_reading,
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


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_macos_discovery_requires_active_noninteractive_admin_permission():
    denied = discover_power_source(
        "Darwin", which_fn=lambda name: f"/usr/bin/{name}",
        run_fn=lambda *_args, **_kwargs: Result(1, stderr="identity hidden"),
        is_file_fn=lambda _path: True,
    )
    assert denied == PowerAvailability(
        False, "powermetrics", "processor_package",
        "administrator permission is not active; run sudo -v before the benchmark",
    )
    available = discover_power_source(
        "Darwin", which_fn=lambda name: f"/usr/bin/{name}",
        run_fn=lambda *_args, **_kwargs: Result(), is_file_fn=lambda _path: True,
    )
    assert available == PowerAvailability(
        True, "powermetrics", "processor_package", location="/usr/bin/powermetrics",
    )


def test_macos_discovery_reports_missing_tool_and_check_failure_without_raising():
    assert discover_power_source(
        "Darwin", which_fn=lambda _name: None, is_file_fn=lambda _path: False,
    ).reason == "powermetrics is not installed"

    def failed_run(*_args, **_kwargs):
        raise OSError("private path and identity")

    status = discover_power_source(
        "Darwin", which_fn=lambda name: f"/usr/bin/{name}", run_fn=failed_run,
        is_file_fn=lambda _path: True,
    )
    assert status.reason == "power permission check failed"


def test_nvidia_and_rocm_discovery_probe_the_counter_not_only_the_executable():
    nvidia = discover_power_source(
        "Linux", which_fn=lambda name: "/bin/nvidia-smi" if name == "nvidia-smi" else None,
        run_fn=lambda *_args, **_kwargs: Result(stdout="125.5\n"),
    )
    assert nvidia == PowerAvailability(
        True, "nvidia-smi", "accelerator", location="/bin/nvidia-smi",
    )
    denied = discover_power_source(
        "Linux", which_fn=lambda name: "/bin/nvidia-smi" if name == "nvidia-smi" else None,
        run_fn=lambda *_args, **_kwargs: Result(1, stderr="permission denied: serial-123"),
    )
    assert denied.reason == "GPU power counters are unreadable"
    rocm = discover_power_source(
        "Linux", which_fn=lambda name: "/bin/rocm-smi" if name == "rocm-smi" else None,
        run_fn=lambda *_args, **_kwargs: Result(stdout=fixture("rocm-smi.json")),
    )
    assert rocm.source == "rocm-smi"
    assert rocm.available is True


def test_rapl_discovery_distinguishes_permission_denied_from_unsupported(monkeypatch, tmp_path):
    counter = tmp_path / "energy_uj"
    counter.write_text("1", encoding="utf-8")
    monkeypatch.setattr("scripts.runtime.telemetry.os.access", lambda *_args: False)
    denied = discover_power_source("Linux", which_fn=lambda _name: None, rapl_paths=[counter])
    assert denied == PowerAvailability(
        False, "rapl", "cpu_package", "RAPL energy counter permission is denied",
    )
    unsupported = discover_power_source("Linux", which_fn=lambda _name: None, rapl_paths=[])
    assert unsupported.source == "unsupported"
    assert unsupported.scope == "unknown"


def test_power_query_uses_discovered_source_and_never_returns_zero_for_failure():
    status = PowerAvailability(True, "nvidia-smi", "accelerator", location="/bin/nvidia-smi")
    assert query_power_reading(
        status, run_fn=lambda *_args, **_kwargs: Result(stdout="42\n"),
    ) == PowerReading(42, "nvidia-smi", "accelerator")
    assert query_power_reading(
        status, run_fn=lambda *_args, **_kwargs: Result(1, stdout="0\n"),
    ) is None
    assert query_power_reading(PowerAvailability(
        False, "nvidia-smi", "accelerator", "denied",
    )) is None
