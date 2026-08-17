import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

from scripts.release.telemetry_qualification import (
    analyze_manifest, analyze_pairs, analyze_sustained_pairs, extract_case_metrics,
    extract_sustained_metrics, metric_impacts, percentile, validate_temperature_mode,
)


def metrics(ttft=1.0, throughput=100.0, wall=2.0):
    return {"ttft": ttft, "throughput": throughput, "wall": wall}


def pairs(on=None, count=20):
    on = on or metrics(1.01, 99.5, 2.01)
    return [
        {"order": "off-on" if index % 2 == 0 else "on-off", "off": metrics(), "on": on}
        for index in range(count)
    ]


def test_percentile_interpolates_and_impacts_penalize_expected_direction():
    assert percentile([0, 10], 0.9) == pytest.approx(9)
    assert metric_impacts(metrics(), metrics(1.02, 99, 2.02)) == pytest.approx({
        "ttft": 2, "throughput": 1, "wall": 1,
    })


def test_screen_passes_at_median_bounds_and_rejects_p90_tail():
    report = analyze_pairs(pairs(on=metrics(1.02, 99, 2.02)))
    assert report["passed"] is True
    tailed = pairs()
    for pair in tailed[-3:]:
        pair["on"] = metrics(1.10, 90, 2.20)
    report = analyze_pairs(tailed)
    assert report["passed"] is False
    assert report["metrics"]["ttft"]["p90_impact_pct"] > 4


def test_ttft_requires_relative_and_duration_bounds_to_fail():
    fast = pairs(on=metrics(0.0295, 100, 2))
    for pair in fast:
        pair["off"] = metrics(0.028, 100, 2)
    report = analyze_pairs(fast)
    assert report["metrics"]["ttft"]["median_impact_pct"] > 2
    assert report["metrics"]["ttft"]["median_impact_sec"] == pytest.approx(0.0015)
    assert report["metrics"]["ttft"]["median_bound_sec"] == 0.002
    assert report["metrics"]["ttft"]["p90_bound_sec"] == 0.004
    assert report["passed"] is True

    slower = pairs(on=metrics(0.031, 100, 2))
    for pair in slower:
        pair["off"] = metrics(0.028, 100, 2)
    assert analyze_pairs(slower)["passed"] is False


def test_ttft_p90_requires_relative_and_duration_bounds_to_fail():
    trial_pairs = pairs(on=metrics(0.028, 100, 2))
    for pair in trial_pairs:
        pair["off"] = metrics(0.028, 100, 2)
    for pair in trial_pairs[-3:]:
        pair["on"] = metrics(0.033, 100, 2)
    report = analyze_pairs(trial_pairs)
    assert report["metrics"]["ttft"]["p90_impact_sec"] > 0.004
    assert report["metrics"]["ttft"]["passed"] is False


@pytest.mark.parametrize("bad", [pairs(count=19), [{**pair, "order": "off-on"} for pair in pairs()]])
def test_screen_rejects_underpowered_or_non_alternating_pairs(bad):
    with pytest.raises(ValueError):
        analyze_pairs(bad)


def test_extracts_case_metrics_and_manifest_paths(tmp_path):
    result = {"llm": {"model": {"2K": {
        "client_ttft_mean_sec": 1.0, "tps_mean": 100,
        "valid_samples": [{"client_wall_sec": 3}, {"client_wall_sec": 1}],
    }}}}
    assert extract_case_metrics(result, "llm", "model", "2K") == metrics(wall=2)
    for index in range(40):
        (tmp_path / f"result-{index}.json").write_text(json.dumps(result))
    manifest = {
        "platform": "test", "interval_sec": 1, "telemetry_mode": "power",
        "source": "powermetrics", "scope": "processor_package",
        "section": "llm", "model": "model", "case": "2K",
        "pairs": [
            {"order": "off-on" if index % 2 == 0 else "on-off",
             "off": f"result-{index * 2}.json", "on": f"result-{index * 2 + 1}.json"}
            for index in range(20)
        ],
    }
    report = analyze_manifest(manifest, tmp_path)
    assert report["pair_count"] == 20
    assert report["passed"] is True
    assert report["telemetry_mode"] == "power"
    assert report["source"] == "powermetrics"
    assert report["scope"] == "processor_package"


def test_extract_rejects_missing_or_nonpositive_metrics():
    with pytest.raises(ValueError, match="lacks"):
        extract_case_metrics({"llm": {"m": {"2K": {"tps_mean": 1}}}}, "llm", "m", "2K")


def sustained_result(retention=0.98, tokens=1000, duration=10):
    return {"sustained": {"model": {
        "series": [{"tokens": tokens, "duration_sec": duration}],
        "analysis": {"retention_ratio": retention},
    }}}


def test_extracts_duration_weighted_sustained_metrics_and_rejects_bad_windows():
    result = sustained_result()
    result["sustained"]["model"]["series"].append({"tokens": 500, "duration_sec": 10})
    assert extract_sustained_metrics(result, "model") == {
        "throughput": 75, "retention_pct": 98,
    }
    bad = sustained_result()
    bad["sustained"]["model"]["series"][0]["duration_sec"] = 0
    with pytest.raises(ValueError, match="invalid duration"):
        extract_sustained_metrics(bad, "model")


def test_sustained_screen_checks_throughput_percent_and_retention_points():
    trial_pairs = [
        {"order": "off-on" if index % 2 == 0 else "on-off",
         "off": {"throughput": 100, "retention_pct": 98},
         "on": {"throughput": 99, "retention_pct": 97}}
        for index in range(20)
    ]
    report = analyze_sustained_pairs(trial_pairs)
    assert report["passed"] is True
    assert report["metrics"]["throughput"]["unit"] == "percent"
    assert report["metrics"]["retention"]["unit"] == "percentage_points"
    for pair in trial_pairs[-3:]:
        pair["on"] = {"throughput": 95, "retention_pct": 94}
    assert analyze_sustained_pairs(trial_pairs)["passed"] is False


def test_manifest_dispatches_to_sustained_observer_screen(tmp_path):
    for index in range(40):
        enabled = index % 2 == 1
        result = sustained_result()
        result.update({
            "run": {"effective_config": {
                "temperature_telemetry": enabled,
                "memory_telemetry": True, "power_telemetry": True,
            }},
            "preflight": ({"temperature": {"available": True}} if enabled else {}),
        })
        (tmp_path / f"result-{index}.json").write_text(
            json.dumps(result), encoding="utf-8",
        )
    manifest = {
        "platform": "test", "interval_sec": 0.5, "telemetry_mode": "temperature",
        "source": {"gpu_die_c": "nvidia-smi"}, "scope": "sensor_channels",
        "section": "sustained", "model": "model", "case": "soak",
        "pairs": [
            {"order": "off-on" if index % 2 == 0 else "on-off",
             "off": f"result-{index * 2}.json", "on": f"result-{index * 2 + 1}.json"}
            for index in range(20)
        ],
    }
    report = analyze_manifest(manifest, tmp_path)
    assert report["passed"] is True
    assert report["section"] == "sustained"
    assert report["metrics"]["retention"]["unit"] == "percentage_points"


def test_temperature_mode_rejects_wrong_identity_or_unavailable_source():
    valid = {
        "run": {"effective_config": {"temperature_telemetry": True,
            "memory_telemetry": True, "power_telemetry": True}},
        "preflight": {"temperature": {"available": True}},
    }
    validate_temperature_mode(valid, True)
    with pytest.raises(ValueError, match="off/on mode"):
        validate_temperature_mode(valid, False)
    valid["preflight"]["temperature"]["available"] = False
    with pytest.raises(ValueError, match="no available"):
        validate_temperature_mode(valid, True)


def test_windows_launcher_delegates_pair_ordering_to_powershell():
    batch = (ROOT / "qual_windows.bat").read_text()
    powershell = (ROOT / "qual_windows.ps1").read_text()
    assert 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File' in batch
    assert '$order = if ($pair % 2 -eq 1) { @("off", "on") } else { @("on", "off") }' in powershell
    assert 'Invoke-BenchmarkCase -Mode $order[0]' in powershell
    assert 'Invoke-BenchmarkCase -Mode $order[1]' in powershell
    assert 'Start-Sleep -Seconds 30' in powershell
