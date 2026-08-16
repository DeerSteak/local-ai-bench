from pathlib import Path

import pytest

from scripts.runtime.telemetry import (
    AmdAdlPowerSource, CaseTelemetry, PollingPowerSource, PowerAvailability, PowerReading,
    PowermetricsPowerSource, TelemetrySample, TelemetrySampler, discover_power_source,
    discover_amd_adl_power,
    add_power_efficiency, derive_run_power_summary, efficiency_per_joule,
    integrate_power_joules, power_block,
    parse_amd_adl_power, parse_nvidia_power, parse_powermetrics_power, parse_rapl_energy_uj,
    parse_rocm_power, power_availability_dict, query_power_reading,
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


def test_amd_adl_power_prefers_asic_then_board_and_rejects_invalid_values():
    sensors = [(0, 0)] * 74
    sensors[23] = (1, 185)
    sensors[73] = (1, 210)
    assert parse_amd_adl_power(sensors) == 185
    sensors[23] = (0, 185)
    assert parse_amd_adl_power(sensors) == 210
    sensors[73] = (1, -1)
    assert parse_amd_adl_power(sensors) is None


class AdlFunction:
    def __init__(self, callback):
        self.callback = callback
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeAdlLibrary:
    def __init__(self, adapter_powers=(120, 80), create_result=0, count_result=0):
        self.destroyed = False
        self.ADL2_Main_Control_Create = AdlFunction(
            lambda _allocator, _connected, context: self._create(context, create_result)
        )
        self.ADL2_Main_Control_Destroy = AdlFunction(self._destroy)
        self.ADL2_Adapter_NumberOfAdapters_Get = AdlFunction(
            lambda _context, count: self._count(count, len(adapter_powers), count_result)
        )
        self.ADL2_New_QueryPMLogData_Get = AdlFunction(
            lambda _context, index, output: self._query(index, output, adapter_powers)
        )

    @staticmethod
    def _create(context, result):
        if result == 0:
            context._obj.value = 123
        return result

    def _destroy(self, _context):
        self.destroyed = True
        return 0

    @staticmethod
    def _count(count, value, result):
        count._obj.value = value
        return result

    @staticmethod
    def _query(index, output, powers):
        output._obj.sensors[23].supported = 1
        output._obj.sensors[23].value = powers[index]
        return 0


def test_windows_amd_adl_source_sums_gpu_power_and_cleans_up_driver_context():
    library = FakeAdlLibrary()
    availability = PowerAvailability(True, "amd-adl", "accelerator", location="atiadlxx.dll")
    source = AmdAdlPowerSource(availability, library_factory=lambda: library)
    source.start()
    assert source.read_watts() == 200
    source.stop()
    assert library.destroyed is True
    assert source.read_watts() is None


def test_windows_amd_adl_source_cleans_up_when_adapter_enumeration_fails():
    library = FakeAdlLibrary(count_result=-1)
    source = AmdAdlPowerSource(
        PowerAvailability(True, "amd-adl", "accelerator", location="atiadlxx.dll"),
        library_factory=lambda: library,
    )
    source.start()
    assert library.destroyed is True
    assert source.read_watts() is None


def test_windows_amd_adl_discovery_distinguishes_missing_driver_from_unreadable_counter():
    assert discover_amd_adl_power(library_factory=lambda: None) is None
    unreadable = discover_amd_adl_power(
        library_factory=lambda: FakeAdlLibrary(adapter_powers=()),
    )
    assert unreadable == PowerAvailability(
        False, "amd-adl", "accelerator", "AMD driver power counters are unreadable",
    )
    available = discover_amd_adl_power(library_factory=lambda: FakeAdlLibrary())
    assert available == PowerAvailability(
        True, "amd-adl", "accelerator", location="atiadlxx.dll",
    )


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


def test_efficiency_block_keeps_unit_work_and_unknown_ratio_auditable():
    source = {"energy_joules": 20, "scope": "accelerator"}
    assert add_power_efficiency(source, "tokens_per_joule", 100) == {
        "energy_joules": 20, "scope": "accelerator",
        "efficiency": {"unit": "tokens_per_joule", "work_count": 100, "per_joule": 5},
    }
    assert "efficiency" not in source
    unknown = add_power_efficiency({"energy_joules": None}, "images_per_joule", 3)
    assert unknown is not None
    assert unknown["efficiency"] == {
        "unit": "images_per_joule", "work_count": 3, "per_joule": None,
    }
    assert add_power_efficiency(None, "tokens_per_joule", 1) is None


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
    assert power_availability_dict(available) == {
        "available": True, "source": "powermetrics", "scope": "processor_package",
        "reason": None,
    }
    assert "/usr/bin" not in str(power_availability_dict(available))


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


def test_windows_discovery_uses_amd_driver_after_nvidia_and_before_linux_sources():
    amd = PowerAvailability(True, "amd-adl", "accelerator", location="atiadlxx.dll")
    assert discover_power_source(
        "Windows", which_fn=lambda _name: None, adl_discovery_fn=lambda: amd,
        rapl_paths=[],
    ) == amd


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


def test_power_block_keeps_idle_separate_and_integrates_only_measured_windows():
    samples = [
        TelemetrySample(0, "idle", power_watts=5),
        TelemetrySample(1, "idle", power_watts=7),
        TelemetrySample(2, "measured:prefill", power_watts=10),
        TelemetrySample(2.5, "measured:prefill", power_watts=14),
        TelemetrySample(3, "measured:decode", power_watts=20),
        TelemetrySample(4.5, "measured:decode", power_watts=24),
    ]
    block = power_block(
        samples, 0.5,
        PowerAvailability(True, "powermetrics", "processor_package", location="tool"), 1,
    )
    assert block["status"] == "recorded"
    assert block["energy_joules"] == pytest.approx(39)
    assert block["idle_baseline_watts"] == 6
    assert block["scope"] == "processor_package"
    assert block["provenance"] == {"interval_sec": 0.5, "failed_samples": 1}
    assert [window["name"] for window in block["windows"]] == [
        "idle", "measured:prefill", "measured:decode",
    ]
    assert block["windows"][0]["energy_joules"] == 6


def test_power_block_records_unavailable_reason_and_never_zero_energy():
    block = power_block(
        [TelemetrySample(0, "measured", power_watts=None)], 0.5,
        PowerAvailability(False, "powermetrics", "processor_package", "permission denied"), 1,
    )
    assert block["status"] == "unavailable"
    assert block["reason"] == "permission denied"
    assert block["energy_joules"] is None
    assert block["mean_watts"] is None


def test_power_block_refuses_partial_multiwindow_energy_and_nonfinite_samples():
    block = power_block([
        TelemetrySample(0, "measured:first", power_watts=10),
        TelemetrySample(1, "measured:first", power_watts=20),
        TelemetrySample(2, "measured:second", power_watts=float("nan")),
    ], 0.5, PowerAvailability(
        True, "nvidia-smi", "accelerator", location="tool",
    ), 1)
    assert block["status"] == "unavailable"
    assert block["energy_joules"] is None
    assert block["reason"] == "insufficient valid samples for energy integration"
    assert block["mean_watts"] == 15


class FakePowerSource:
    def __init__(self, watts=12):
        self.watts = watts
        self.calls = []

    def start(self):
        self.calls.append("start")

    def read_watts(self):
        self.calls.append("read")
        return self.watts

    def stop(self):
        self.calls.append("stop")


def test_shared_sampler_starts_reads_and_stops_power_on_the_memory_timeline():
    source = FakePowerSource()
    sampler = TelemetrySampler(42, interval_sec=100, power_source=source)
    sampler.start()
    sampler.capture()
    samples = sampler.stop()
    assert source.calls[0] == "start"
    assert source.calls[-1] == "stop"
    assert all(sample.power_watts == 12 for sample in samples)
    assert sampler.power_failures == 0


def test_shared_sampler_turns_power_errors_into_unknown_samples():
    source = FakePowerSource()

    def fail():
        raise OSError("sensor failed")

    source.read_watts = fail
    sampler = TelemetrySampler(42, interval_sec=100, power_source=source).start()
    samples = sampler.stop()
    assert samples[0].power_watts is None
    assert sampler.failed_samples == 1
    assert sampler.power_failures == 1


class FakeProcess:
    def __init__(self):
        self.stdout = iter([
            "CPU Power: 1000 mW\n",
            "Combined Power (CPU + GPU + ANE): 2500 mW\n",
        ])
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        return 0

    def kill(self):
        raise AssertionError("clean process should not be killed")


def test_powermetrics_source_uses_one_long_lived_noninteractive_process():
    process = FakeProcess()
    commands = []

    def popen(command, **_kwargs):
        commands.append(command)
        return process

    source = PowermetricsPowerSource(
        PowerAvailability(True, "powermetrics", "processor_package", location="/tool"),
        0.5, popen_fn=popen,
    )
    source.start()
    assert source._reader is not None
    source._reader.join(timeout=1)
    assert source.read_watts() == 2.5
    source.stop()
    assert process.terminated is True
    assert commands == [[
        "/usr/bin/sudo", "-n", "/tool", "--samplers", "cpu_power,gpu_power,ane_power",
        "--sample-rate", "500", "--sample-count", "-1", "--buffer-size", "1",
    ]]


def test_rapl_polling_source_derives_watts_from_counter_delta():
    values = iter(["1000000", "2500000", "500000"])
    times = iter([10.0, 10.5, 11.0])
    source = PollingPowerSource(
        PowerAvailability(True, "rapl", "cpu_package", location="energy_uj"),
        read_text_fn=lambda _path: next(values), monotonic=lambda: next(times),
    )
    source.start()
    assert source.read_watts() is None
    assert source.read_watts() == 3
    assert source.read_watts() is None


def test_case_telemetry_exposes_case_local_power_beside_legacy_memory_result():
    readings = iter([5, 7, 10, 14, 8])

    def sample(timestamp, window):
        return TelemetrySample(timestamp, window, host_ram_used_gb=10,
                               power_watts=next(readings))

    sampler = TelemetrySampler(42, interval_sec=100, sample_fn=sample)
    telemetry = CaseTelemetry(
        sampler=sampler, sources={"host_ram_used_gb": "psutil"},
        power_availability=PowerAvailability(
            True, "powermetrics", "processor_package", location="tool",
        ),
    ).start()
    telemetry.begin_measured()
    memory = telemetry.finish_case()
    telemetry.stop()
    assert memory["summary"]["host_ram_used_gb"]["peak_gb"] == 10
    assert telemetry.last_power is not None
    assert telemetry.last_power["status"] == "recorded"
    assert telemetry.last_power["energy_joules"] is not None


def test_run_power_summary_sums_only_same_scope_case_energy():
    sections = {"llm": {"model": {
        "2K": {"power": {"status": "recorded", "source": "powermetrics",
                           "scope": "processor_package", "energy_joules": 12,
                           "idle_baseline_watts": 4}},
        "8K": {"power": {"status": "recorded", "source": "powermetrics",
                           "scope": "processor_package", "energy_joules": 20,
                           "idle_baseline_watts": 6}},
    }}}
    assert derive_run_power_summary(sections) == {
        "status": "recorded", "reason": None, "scope": "processor_package",
        "source": "powermetrics", "energy_joules": 32, "idle_baseline_watts": 5,
        "recorded_cases": 2, "total_cases": 2,
    }


def test_run_power_summary_refuses_mixed_scope_total_and_preserves_unavailable():
    mixed = derive_run_power_summary({
        "llm": {"a": {"power": {"scope": "accelerator", "source": "nvidia-smi",
                                    "energy_joules": 8}}},
        "embeddings": {"b": {"power": {"scope": "cpu_package", "source": "rapl",
                                          "energy_joules": 2}}},
    })
    assert mixed is not None
    assert mixed["scope"] == "mixed"
    assert mixed["energy_joules"] is None
    assert derive_run_power_summary({"llm": {"legacy": {"tps_mean": 4}}}) is None
    unavailable = derive_run_power_summary({"llm": {"a": {"power": {
        "status": "unavailable", "source": "powermetrics", "scope": "processor_package",
        "energy_joules": None, "reason": "permission denied",
    }}}})
    assert unavailable is not None
    assert unavailable["status"] == "unavailable"
    assert unavailable["energy_joules"] is None
    assert unavailable["reason"] == "permission denied"
    invalid = derive_run_power_summary({"llm": {"a": {"power": {
        "source": "powermetrics", "scope": "processor_package",
        "energy_joules": float("nan"), "idle_baseline_watts": True,
    }}}})
    assert invalid is not None
    assert invalid["energy_joules"] is None
    assert invalid["idle_baseline_watts"] is None
