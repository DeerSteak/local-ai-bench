import json

from scripts.results.result_history import (
    compare_results, discover_results, extract_comparable_metrics, filter_results,
    summarize_result,
)


def result(*, hostname="system", started="2026-01-01T00:00:00Z", tps=50.0):
    settings = {
        "methodology_profile": "neutral-v1", "runs": 3, "cpu_only": False,
        "effective_optimizations": ["llamacpp:flash_attention=on"],
    }
    return {
        "version": "4.1", "engine": "llamacpp", "profile": {"hostname": hostname},
        "run": {"started_at": started, "status": "complete", "plan": {"effective_config": settings},
                "stages": {"llm": {"models_with_results": 1}}},
        "llm": {"model": {"2K": {"tps_mean": tps, "ttft_mean_sec": 0.2}}},
        "embeddings": {"embed": {"chunks_per_sec_mean": 100.0}},
        "images": {"flux": {"resolutions": {"1024x1024": {"sec_per_image_mean": 8.0}}}},
        "mcq": {"model": {"accuracy_pct": 75.0}},
    }


def test_summary_and_discovery_sort_results_and_report_malformed_files(tmp_path):
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"
    older.write_text(json.dumps(result(started="2026-01-01T00:00:00Z")), encoding="utf-8")
    newer.write_text(json.dumps(result(hostname="new", started="2026-02-01T00:00:00Z")), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    entries, skipped = discover_results(tmp_path)
    assert [entry["system"] for entry in entries] == ["new", "system"]
    assert entries[0]["models_with_results"] == 1
    assert skipped[0]["path"].endswith("bad.json")


def test_history_filter_combines_query_status_and_engine():
    entries = [
        summarize_result(result(hostname="Alpha"), "alpha.json"),
        dict(summarize_result(result(hostname="Beta"), "beta.json"), status="interrupted"),
    ]
    assert [item["system"] for item in filter_results(entries, query="alp")] == ["Alpha"]
    assert [item["system"] for item in filter_results(entries, status="interrupted")] == ["Beta"]
    assert filter_results(entries, engine="other") == []


def test_metric_extraction_uses_named_supported_evidence_only():
    metrics = extract_comparable_metrics(result())
    assert metrics["llm/model/2K/tps_mean"] == 50.0
    assert metrics["embeddings/embed/chunks_per_sec_mean"] == 100.0
    assert metrics["images/flux/1024x1024/sec_per_image_mean"] == 8.0
    assert metrics["mcq/model/accuracy_pct"] == 75.0


def test_comparison_reports_exact_deltas_for_compatible_results():
    comparison = compare_results(result(tps=50.0), result(tps=55.0))
    assert comparison["compatible"] is True
    row = next(item for item in comparison["rows"] if item["metric"] == "llm/model/2K/tps_mean")
    assert row == {"metric": "llm/model/2K/tps_mean", "baseline": 50.0,
                   "candidate": 55.0, "delta": 5.0, "percent_change": 10.0}


def test_comparison_blocks_different_or_unrecorded_methodology_and_keeps_missing_rows():
    baseline = result()
    candidate = result()
    candidate["run"]["plan"]["effective_config"]["runs"] = 5
    candidate["llm"]["other"] = candidate["llm"].pop("model")
    comparison = compare_results(baseline, candidate)
    assert comparison["compatible"] is False
    assert "effective_config" in comparison["incompatible_fields"]
    assert any(row["baseline"] is None for row in comparison["rows"])
    del baseline["run"]["plan"]["effective_config"]["methodology_profile"]
    assert "unrecorded_methodology" in compare_results(baseline, result())["incompatible_fields"]
