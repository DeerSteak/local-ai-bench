from pathlib import Path

from scripts.runtime.telemetry import (
    PollingTemperatureSource, TemperatureAvailability, TemperatureReading,
    TelemetrySample, TelemetrySampler, discover_temperature_source,
    parse_hwmon_temperature, parse_nvidia_temperatures, parse_rocm_temperatures,
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


def test_temperature_block_preserves_aligned_timestamps_and_missing_channels():
    samples = [
        TelemetrySample(0, "measured:0", power_watts=100, gpu_die_c=60),
        TelemetrySample(1.25, "measured:0", power_watts=95, gpu_die_c=70),
    ]
    block = temperature_block(
        samples, 0.5, TemperatureAvailability(True, {"gpu_die_c": "nvidia-smi"}),
        {"cpu_package_c": 2, "gpu_hotspot_c": 2},
    )
    assert block["status"] == "recorded"
    assert block["windows"][0]["duration_sec"] == 1.25
    assert block["windows"][0]["channels"]["gpu_die_c"] == {
        "peak_c": 70, "mean_c": 65, "final_c": 70, "valid_samples": 2,
    }
    assert block["windows"][0]["samples"] == [
        {"timestamp_sec": 0, "cpu_package_c": None, "gpu_die_c": 60,
         "gpu_hotspot_c": None},
        {"timestamp_sec": 1.25, "cpu_package_c": None, "gpu_die_c": 70,
         "gpu_hotspot_c": None},
    ]


def test_sampler_counts_each_missing_temperature_channel_without_losing_memory():
    class Source:
        def read(self):
            return TemperatureReading(gpu_die_c=64)

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
        "cpu_package_c": 1, "gpu_die_c": 0, "gpu_hotspot_c": 1,
    }
