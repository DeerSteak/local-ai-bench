import json

import pytest

from scripts.release.telemetry_qualification import (
    analyze_manifest, analyze_pairs, extract_case_metrics, metric_impacts, percentile,
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
        "platform": "test", "interval_sec": 1, "section": "llm", "model": "model", "case": "2K",
        "pairs": [
            {"order": "off-on" if index % 2 == 0 else "on-off",
             "off": f"result-{index * 2}.json", "on": f"result-{index * 2 + 1}.json"}
            for index in range(20)
        ],
    }
    report = analyze_manifest(manifest, tmp_path)
    assert report["pair_count"] == 20
    assert report["passed"] is True


def test_extract_rejects_missing_or_nonpositive_metrics():
    with pytest.raises(ValueError, match="lacks"):
        extract_case_metrics({"llm": {"m": {"2K": {"tps_mean": 1}}}}, "llm", "m", "2K")
