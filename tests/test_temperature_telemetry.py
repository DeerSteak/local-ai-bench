from pathlib import Path

from scripts.runtime.telemetry import (
    AppleHidTemperatureSource, NvidiaTelemetryReading, PollingPowerSource,
    PollingTemperatureSource, PowerAvailability,
    TemperatureAvailability, TemperatureReading,
    TelemetrySample, TelemetrySampler, discover_temperature_source,
    aggregate_apple_hid_temperatures,
    memory_block, power_block,
    parse_hwmon_temperature, parse_nvidia_telemetry, parse_nvidia_temperatures,
    parse_rocm_temperatures,
    temperature_availability_dict, temperature_block,
)


class Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_nvidia_temperature_parser_uses_hottest_die_and_rejects_partial_output():
    assert parse_nvidia_temperatures("62\n71 C\nN/A\n") == TemperatureReading(gpu_die_c=71)
    assert parse_nvidia_temperatures("N/A\n[Not Supported]\n") is None
    assert parse_nvidia_temperatures("201\nnan\n-1\n") is None


def test_nvidia_combined_parser_aggregates_devices_and_preserves_partial_channels():
    assert parse_nvidia_telemetry("1024, 8192, 120.5, 65\n2048, 16384, 80, 72\n") == \
        NvidiaTelemetryReading(3, 24, 200.5, 72)
    assert parse_nvidia_telemetry("N/A, N/A, 50, N/A\n") == \
        NvidiaTelemetryReading(power_watts=50)
    assert parse_nvidia_telemetry("N/A, N/A, N/A, N/A\n") is None


def test_rocm_temperature_parser_separates_die_and_hotspot_across_devices():
    output = """{
      "card0": {
        "Temperature (Sensor edge) (C)": "54.0",
        "Temperature (Sensor junction) (C)": "67.0"
      },
      "card1": {
        "Temperature (Sensor edge) (C)": "58.0",
        "Temperature (Sensor junction) (C)": "73.0"
      }
    }"""
    assert parse_rocm_temperatures(output) == TemperatureReading(
        gpu_die_c=58, gpu_hotspot_c=73,
    )
    assert parse_rocm_temperatures('{"card0": {"Temperature": "N/A"}}') is None
    assert parse_rocm_temperatures('{"card0":') is None
    assert parse_rocm_temperatures("[]") is None


def test_hwmon_parser_accepts_package_labels_and_rejects_unrelated_or_invalid_sensors():
    assert parse_hwmon_temperature("Package id 0", "62500") == ("cpu_package_c", 62.5)
    assert parse_hwmon_temperature("Tctl", "71000") == ("cpu_package_c", 71)
    assert parse_hwmon_temperature("Tdie", "68000") == ("cpu_package_c", 68)
    assert parse_hwmon_temperature("Core 0", "59000") is None
    assert parse_hwmon_temperature("Tctl", "permission denied") is None
    assert parse_hwmon_temperature("Tctl", "201000") is None


def test_apple_hid_aggregation_combines_named_sensor_families_as_soc_package():
    assert aggregate_apple_hid_temperatures([
        ("pACC MTR Temp Sensor0", 50),
        ("eACC MTR Temp Sensor1", 46),
        ("GPU MTR Temp Sensor0", 44),
        ("gas gauge battery", 31),
        ("PMU tcal", 80),
    ]) == TemperatureReading(soc_package_c=140 / 3)
    assert aggregate_apple_hid_temperatures([
        ("PMU tdie1", 40), ("PMU tdie2", 42), ("NAND CH0 temp", 35),
    ]) == TemperatureReading(soc_package_c=41)
    assert aggregate_apple_hid_temperatures([
        ("PMU tdie1", 0), ("GPU MTR Temp Sensor0", 201),
    ]) is None


def test_darwin_discovery_probes_apple_hid_and_preserves_available_channels():
    class Reader:
        def read(self):
            return TemperatureReading(soc_package_c=48)

    status = discover_temperature_source(
        "Darwin", machine_name="arm64", which_fn=lambda _name: None,
        apple_reader_factory=Reader,
    )
    assert status == TemperatureAvailability(
        True, {"soc_package_c": "apple-hid"},
        locations={},
    )


def test_darwin_discovery_rejects_empty_or_unsupported_apple_hid_probe():
    class EmptyReader:
        def read(self):
            return None

    unavailable = discover_temperature_source(
        "Darwin", machine_name="arm64", which_fn=lambda _name: None,
        apple_reader_factory=EmptyReader,
    )
    assert unavailable.available is False
    assert unavailable.reason == "no supported temperature source was detected"

    def unexpected_reader():
        raise AssertionError("Intel macOS must not probe Apple Silicon HID sensors")

    intel = discover_temperature_source(
        "Darwin", machine_name="x86_64", which_fn=lambda _name: None,
        apple_reader_factory=unexpected_reader,
    )
    assert intel.available is False


def test_apple_hid_source_lifecycle_turns_reader_failure_into_missing_evidence():
    calls = []

    class Reader:
        def read(self):
            calls.append("read")
            if len(calls) == 1:
                return TemperatureReading(soc_package_c=47)
            raise OSError("sensor disappeared")

    source = AppleHidTemperatureSource(
        TemperatureAvailability(True, {"soc_package_c": "apple-hid"}),
        reader_factory=Reader,
    )
    source.start()
    assert source.read() == TemperatureReading(soc_package_c=47)
    assert source.read() == TemperatureReading()
    source.stop()
    assert source.read() == TemperatureReading()


def test_sampler_starts_and_stops_temperature_source():
    calls = []

    class Source:
        def start(self):
            calls.append("start")

        def read(self):
            calls.append("read")
            return TemperatureReading(soc_package_c=45)

        def stop(self):
            calls.append("stop")

    sampler = TelemetrySampler(42, interval_sec=100, temperature_source=Source()).start()
    samples = sampler.stop()
    assert calls[0] == "start"
    assert "read" in calls
    assert calls[-1] == "stop"
    assert samples[0].soc_package_c == 45


def write_hwmon(root: Path, device: str, name: str, label: str, value: str) -> None:
    path = root / device
    path.mkdir()
    (path / "name").write_text(name, encoding="utf-8")
    (path / "temp1_label").write_text(label, encoding="utf-8")
    (path / "temp1_input").write_text(value, encoding="utf-8")


def test_linux_discovery_combines_hwmon_cpu_and_nvidia_gpu_without_exposing_paths(tmp_path):
    write_hwmon(tmp_path, "hwmon0", "k10temp", "Tctl", "65000")
    status = discover_temperature_source(
        "Linux", hwmon_root=tmp_path,
        which_fn=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
        run_fn=lambda *_args, **_kwargs: Result(stdout="57\n"),
    )
    assert status.available is True
    assert status.sources == {"cpu_package_c": "hwmon", "gpu_die_c": "nvidia-smi"}
    assert status.locations is not None
    assert temperature_availability_dict(status) == {
        "available": True,
        "sources": {"cpu_package_c": "hwmon", "gpu_die_c": "nvidia-smi"},
        "reason": None,
    }
    assert str(tmp_path) not in str(temperature_availability_dict(status))


def test_discovery_retains_cpu_channel_when_accelerator_probe_fails(tmp_path):
    write_hwmon(tmp_path, "hwmon0", "coretemp", "Package id 0", "61000")
    status = discover_temperature_source(
        "Linux", hwmon_root=tmp_path,
        which_fn=lambda name: "/bin/nvidia-smi" if name == "nvidia-smi" else None,
        run_fn=lambda *_args, **_kwargs: Result(1, stderr="permission denied: gpu serial"),
    )
    assert status.available is True
    assert status.sources == {"cpu_package_c": "hwmon"}


def test_discovery_falls_back_to_rocm_when_nvidia_probe_fails(tmp_path):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[0] == "/bin/nvidia-smi":
            return Result(1, stderr="driver unavailable")
        return Result(stdout=(
            '{"card0":{"Temperature (Sensor edge) (C)":58,'
            '"Temperature (Sensor junction) (C)":74}}'
        ))

    status = discover_temperature_source(
        "Linux", hwmon_root=tmp_path,
        which_fn=lambda name: f"/bin/{name}" if name in {"nvidia-smi", "rocm-smi"} else None,
        run_fn=run,
    )

    assert [command[0] for command in commands] == ["/bin/nvidia-smi", "/bin/rocm-smi"]
    assert status.sources == {"gpu_die_c": "rocm-smi", "gpu_hotspot_c": "rocm-smi"}


def test_discovery_reports_unavailable_when_no_supported_channel_exists(tmp_path):
    write_hwmon(tmp_path, "hwmon0", "acpitz", "temp1", "40000")
    status = discover_temperature_source(
        "Linux", hwmon_root=tmp_path, which_fn=lambda _name: None,
    )
    assert status == TemperatureAvailability(
        False, {}, "no supported temperature source was detected",
    )


def test_polling_source_reads_cpu_and_rocm_channels_from_one_capture():
    status = TemperatureAvailability(
        True,
        {"cpu_package_c": "hwmon", "gpu_die_c": "rocm-smi", "gpu_hotspot_c": "rocm-smi"},
        locations={"cpu_package_c": "/sys/tctl", "gpu_die_c": "/bin/rocm-smi",
                   "gpu_hotspot_c": "/bin/rocm-smi"},
    )
    source = PollingTemperatureSource(
        status, read_text_fn=lambda _path: "66000",
        run_fn=lambda *_args, **_kwargs: Result(stdout=(
            '{"card0":{"Temperature (Sensor edge) (C)":55,'
            '"Temperature (Sensor junction) (C)":72}}'
        )),
    )
    assert source.read() == TemperatureReading(66, 55, 72)


def test_memory_power_and_temperature_blocks_preserve_one_timeline():
    samples = [
        TelemetrySample(
            0, "measured:0", host_ram_used_gb=8, process_rss_gb=2,
            power_watts=100, gpu_die_c=60,
        ),
        TelemetrySample(
            1.25, "measured:0", host_ram_used_gb=9, process_rss_gb=3,
            power_watts=95, gpu_die_c=None,
        ),
    ]
    memory = memory_block(
        samples, 0.5, 0, {}, {
            "host_ram_used_gb": "psutil", "process_rss_gb": "psutil",
        },
    )
    power = power_block(
        samples, 0.5, PowerAvailability(True, "nvidia-smi", "accelerator"), 0,
    )
    temperature = temperature_block(
        samples, 0.5, TemperatureAvailability(True, {"gpu_die_c": "nvidia-smi"}),
        {"gpu_die_c": 1},
    )

    timelines = [
        [sample["timestamp_sec"] for sample in block["windows"][0]["samples"]]
        for block in (memory, power, temperature)
    ]
    assert timelines == [[0, 1.25]] * 3
    assert [block["windows"][0]["sample_count"]
            for block in (memory, power, temperature)] == [2, 2, 2]
    assert [block["windows"][0]["duration_sec"]
            for block in (memory, power, temperature)] == [1.25, 1.25, 1.25]
    assert temperature["status"] == "recorded"
    assert temperature["windows"][0]["channels"]["gpu_die_c"] == {
        "peak_c": 60, "mean_c": 60, "final_c": 60, "valid_samples": 1,
    }
    assert temperature["windows"][0]["samples"] == [
        {"timestamp_sec": 0, "soc_package_c": None, "cpu_package_c": None, "gpu_die_c": 60,
         "gpu_hotspot_c": None},
        {"timestamp_sec": 1.25, "soc_package_c": None, "cpu_package_c": None, "gpu_die_c": None,
         "gpu_hotspot_c": None},
    ]
    assert temperature["provenance"]["channels"] == {
        "soc_package_c": {"source": "unsupported", "failed_samples": 0},
        "cpu_package_c": {"source": "unsupported", "failed_samples": 0},
        "gpu_die_c": {"source": "nvidia-smi", "failed_samples": 1},
        "gpu_hotspot_c": {"source": "unsupported", "failed_samples": 0},
    }


def test_temperature_block_retains_failures_for_discovered_channels():
    block = temperature_block(
        [TelemetrySample(0, "measured", cpu_package_c=None)],
        0.5, TemperatureAvailability(True, {"cpu_package_c": "hwmon"}),
        {"cpu_package_c": 1, "gpu_die_c": 1},
    )
    assert block["provenance"]["channels"]["cpu_package_c"] == {
        "source": "hwmon", "failed_samples": 1,
    }
    assert block["provenance"]["channels"]["gpu_die_c"] == {
        "source": "unsupported", "failed_samples": 0,
    }


def test_sampler_counts_each_missing_temperature_channel_without_losing_memory():
    class Source:
        def start(self):
            pass

        def read(self):
            return TemperatureReading(gpu_die_c=64)

        def stop(self):
            pass

    sampler = TelemetrySampler(
        42, temperature_source=Source(),
        sample_fn=lambda timestamp, window: TelemetrySample(
            timestamp, window, host_ram_used_gb=10, gpu_die_c=64,
        ),
    )
    sampler._started_at = 0
    captured = sampler.capture()
    assert captured.host_ram_used_gb == 10
    assert captured.gpu_die_c == 64
    assert sampler.temperature_failures == {
        "soc_package_c": 1, "cpu_package_c": 1, "gpu_die_c": 0, "gpu_hotspot_c": 1,
    }


def test_sampler_uses_one_nvidia_capture_for_memory_power_and_temperature(monkeypatch):
    calls = []
    availability = TemperatureAvailability(
        True, {"gpu_die_c": "nvidia-smi"}, locations={"gpu_die_c": "/bin/nvidia-smi"},
    )
    power = PollingPowerSource(
        PowerAvailability(True, "nvidia-smi", "accelerator", location="/bin/nvidia-smi"),
        run_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("extra query")),
    )
    temperature = PollingTemperatureSource(
        availability,
        run_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("extra query")),
    )
    monkeypatch.setattr("scripts.runtime.telemetry.system_memory_usage", lambda: (8, 16))
    monkeypatch.setattr("scripts.runtime.telemetry.process_resource_usage", lambda _pid: (1, 2))
    sampler = TelemetrySampler(
        42, power_source=power, temperature_source=temperature,
        memory_sources={"accelerator_memory_used_gb": "nvidia-smi"},
        nvidia_query_fn=lambda: calls.append("capture") or NvidiaTelemetryReading(3, 24, 200, 72),
    )

    sample = sampler._sample(0, "measured")

    assert calls == ["capture"]
    assert sample.accelerator_memory_used_gb == 3
    assert sample.accelerator_memory_total_gb == 24
    assert sample.power_watts == 200
    assert sample.gpu_die_c == 72
