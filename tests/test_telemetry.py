import time
import threading

import pytest

from scripts.runtime import config
from scripts.runtime.telemetry import (
    CaseTelemetry, TelemetrySample, TelemetrySampler, calculate_headroom,
    default_memory_sources, derive_run_memory_summary, memory_block, memory_ceiling_gb,
    query_sampler_vram_usage,
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


def test_memory_block_records_provenance_and_unknown_headroom():
    block = memory_block(
        [sample(0, "idle", host=10), sample(1, "measured", host=12)],
        0.5, 2, {"host_ram_used_gb": 1}, {"host_ram_used_gb": "psutil"},
    )
    assert [window["name"] for window in block["windows"]] == ["idle", "measured"]
    assert block["windows"][0]["samples"] == [{
        "timestamp_sec": 0,
        "host_ram_used_gb": 10,
        "process_rss_gb": None,
        "accelerator_memory_used_gb": None,
        "accelerator_memory_total_gb": None,
    }]
    assert block["windows"][1]["samples"][0]["timestamp_sec"] == 1
    assert block["summary"]["host_ram_used_gb"]["peak_gb"] == 12
    assert block["headroom"] == {
        "absolute_gb": None, "fraction": None, "state": "unknown",
        "basis_channel": "process_rss_gb",
    }
    assert block["provenance"]["failed_samples"] == 2
    assert block["provenance"]["channels"]["host_ram_used_gb"] == {
        "source": "psutil", "failed_samples": 1,
    }
    assert block["provenance"]["channels"]["process_rss_gb"] == {
        "source": "unsupported", "failed_samples": 0,
    }


def test_memory_block_uses_process_rss_for_unified_memory_headroom():
    block = memory_block(
        [sample(0, "measured", host=14, rss=4)], 0.5, 0, {},
        {"host_ram_used_gb": "psutil", "process_rss_gb": "psutil",
         "accelerator_memory_used_gb": "unsupported"}, 16,
    )
    assert block["headroom"] == {
        "absolute_gb": 12, "fraction": 0.75, "state": "comfortable",
        "basis_channel": "process_rss_gb",
    }


def test_memory_block_does_not_hide_failed_accelerator_reading_with_rss():
    block = memory_block(
        [sample(0, "measured", host=14, rss=4)], 0.5, 0, {},
        {"process_rss_gb": "psutil", "accelerator_memory_used_gb": "nvidia-smi"}, 16,
    )
    assert block["headroom"] == {
        "absolute_gb": None, "fraction": None, "state": "unknown",
        "basis_channel": "accelerator_memory_used_gb",
    }


def test_run_summary_reports_each_peak_and_tightest_case():
    sections = {"llm": {"model": {
        "2K": {"memory": {
            "case_id": "case-a", "summary": {
                "process_rss_gb": {"peak_gb": 4}, "host_ram_used_gb": {"peak_gb": 20},
            }, "headroom": {"absolute_gb": 8, "fraction": 0.5, "state": "comfortable",
                            "basis_channel": "process_rss_gb"},
        }},
        "8K": {"memory": {
            "case_id": "case-b", "summary": {
                "process_rss_gb": {"peak_gb": 7}, "host_ram_used_gb": {"peak_gb": 25},
            }, "headroom": {"absolute_gb": 2, "fraction": 0.125, "state": "tight",
                            "basis_channel": "process_rss_gb"},
        }},
    }}}
    assert derive_run_memory_summary(sections) == {
        "channels": {
            "process_rss_gb": {"peak_gb": 7}, "host_ram_used_gb": {"peak_gb": 25},
        },
        "tightest_headroom": {
            "absolute_gb": 2, "fraction": 0.125, "state": "tight",
            "basis_channel": "process_rss_gb",
            "case_id": "case-b", "case_path": "llm/model/8K",
        },
    }
    assert derive_run_memory_summary({"llm": {"legacy": {"2K": {"tps_mean": 4}}}}) is None


def test_run_summary_finds_memory_nested_in_native_entry_lists():
    sections = {"llamabench": {"model": {"prefill_entries": [{
        "n_prompt": 512,
        "memory": {
            "case_id": "native-a",
            "summary": {"process_rss_gb": {"peak_gb": 6}},
            "headroom": {"absolute_gb": 3, "fraction": 0.2, "state": "comfortable"},
        },
    }]}}}
    summary = derive_run_memory_summary(sections)
    assert summary == {
        "channels": {"process_rss_gb": {"peak_gb": 6}},
        "tightest_headroom": {
            "absolute_gb": 3, "fraction": 0.2, "state": "comfortable",
            "case_id": "native-a", "case_path": "llamabench/model/prefill_entries/0",
        },
    }


def test_memory_ceiling_preserves_per_gpu_reserve():
    result = type("Result", (), {"returncode": 0, "stdout": "16384\n8192\n"})()
    ceiling = memory_ceiling_gb(
        {"accelerator_memory_total_gb": "nvidia-smi"},
        run_fn=lambda *_args, **_kwargs: result,
        which_fn=lambda _name: "/usr/bin/nvidia-smi",
    )
    assert ceiling == 22


def test_macos_uses_host_pool_without_claiming_separate_accelerator_counters(monkeypatch):
    monkeypatch.setattr("scripts.runtime.telemetry.platform.system", lambda: "Darwin")
    sources = default_memory_sources(which_fn=lambda _name: None)
    assert sources == {
        "host_ram_used_gb": "psutil",
        "process_rss_gb": "psutil",
        "accelerator_memory_used_gb": "unsupported",
        "accelerator_memory_total_gb": "unsupported",
    }


def test_sampler_vram_query_aggregates_nvidia_devices_and_rejects_bad_output():
    result = type("Result", (), {"returncode": 0, "stdout": "1024, 8192\n2048, 16384\n"})()
    assert query_sampler_vram_usage(
        run_fn=lambda *_args, **_kwargs: result,
        which_fn=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    ) == (3, 24)
    bad = type("Result", (), {"returncode": 0, "stdout": "not memory"})()
    assert query_sampler_vram_usage(
        run_fn=lambda *_args, **_kwargs: bad,
        which_fn=lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    ) is None


def test_sampler_vram_query_normalizes_rocm_bytes():
    result = type("Result", (), {"returncode": 0, "stdout": (
        '{"card0":{"VRAM Total Used Memory (B)":1073741824,'
        '"VRAM Total Memory (B)":8589934592}}'
    )})()
    assert query_sampler_vram_usage(
        run_fn=lambda *_args, **_kwargs: result,
        which_fn=lambda name: "/usr/bin/rocm-smi" if name == "rocm-smi" else None,
    ) == (1, 8)


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
    assert sampler.channel_failures["process_rss_gb"] == len(samples)


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


def test_concurrent_boundary_capture_cannot_append_a_later_window_first():
    entered = threading.Event()
    release = threading.Event()

    def blocking_sample(timestamp, window):
        if window == "idle":
            entered.set()
            release.wait(timeout=1)
        return sample(timestamp, window, rss=2)

    sampler = TelemetrySampler(42, interval_sec=100, sample_fn=blocking_sample)
    sampler._started_at = time.monotonic()
    first = threading.Thread(target=sampler.capture)
    first.start()
    assert entered.wait(timeout=1)
    sampler.set_window("model_load")
    second = threading.Thread(target=sampler.capture)
    second.start()
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert [item.window for item in sampler.samples] == ["idle", "model_load"]
    assert [item.timestamp_sec for item in sampler.samples] == sorted(
        item.timestamp_sec for item in sampler.samples
    )


def test_case_telemetry_does_not_reuse_prior_case_measurements():
    sampler = TelemetrySampler(
        42, interval_sec=10,
        sample_fn=lambda timestamp, window: sample(timestamp, window, rss=2),
    )
    telemetry = CaseTelemetry(sampler=sampler, sources={"process_rss_gb": "psutil"}).start()
    telemetry.begin_model_load()
    telemetry.begin_measured("measured:prefill")
    first = telemetry.finish_case()
    telemetry.begin_model_load()
    telemetry.begin_measured("measured:decode")
    second = telemetry.finish_case()
    telemetry.stop()
    assert [window["name"] for window in first["windows"]] == [
        "idle", "model_load", "measured:prefill",
    ]
    assert [window["name"] for window in second["windows"]] == [
        "idle", "model_load", "measured:decode",
    ]


def test_case_telemetry_uses_one_sample_snapshot_for_slice_and_cursor():
    class CountingSampler(TelemetrySampler):
        sample_reads = 0

        @property
        def samples(self):
            self.sample_reads += 1
            return super().samples

    sampler = CountingSampler(
        42, interval_sec=100,
        sample_fn=lambda timestamp, window: sample(timestamp, window, rss=2),
    )
    telemetry = CaseTelemetry(sampler=sampler, sources={"process_rss_gb": "psutil"}).start()
    telemetry.begin_measured()
    telemetry.finish_case()
    assert sampler.sample_reads == 1
    telemetry.stop()


def test_case_telemetry_failure_provenance_is_case_local():
    counter = {"value": 0}

    def changing_sample(timestamp, window):
        counter["value"] += 1
        return sample(timestamp, window, host=10 if counter["value"] != 2 else None)

    sampler = TelemetrySampler(42, interval_sec=100, sample_fn=changing_sample)
    telemetry = CaseTelemetry(
        sampler=sampler,
        sources={"host_ram_used_gb": "psutil", "process_rss_gb": "unsupported",
                 "accelerator_memory_used_gb": "unsupported",
                 "accelerator_memory_total_gb": "unsupported"},
    ).start()
    first = telemetry.finish_case()
    second = telemetry.finish_case()
    telemetry.stop()
    assert first["provenance"]["channels"]["host_ram_used_gb"]["failed_samples"] == 1
    assert second["provenance"]["channels"]["host_ram_used_gb"]["failed_samples"] == 0


@pytest.mark.parametrize("interval", [0, -0.1])
def test_sampler_rejects_nonpositive_intervals(interval):
    with pytest.raises(ValueError, match="positive"):
        TelemetrySampler(42, interval_sec=interval)
