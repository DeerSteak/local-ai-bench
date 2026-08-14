import json
from pathlib import Path

import pytest

from scripts.results.result_history import (
    compare_results, delete_multiple_run_artifacts, delete_run_artifacts,
    discover_results, existing_run_artifacts,
    extract_comparable_metrics, filter_results, run_artifact_paths, summarize_result,
)


def test_delete_multiple_runs_removes_each_exact_artifact_set(tmp_path):
    first = tmp_path / "results_first.json"
    second = tmp_path / "results_second.json"
    neighbor = tmp_path / "results_neighbor.json"
    log = tmp_path / "log_first.txt"
    for path in (first, second, neighbor, log):
        path.touch()

    removed, failures = delete_multiple_run_artifacts([first, second], tmp_path)

    assert not failures
    assert set(removed) == {first, second, log}
    assert neighbor.exists()


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


def test_run_artifact_paths_cover_run_journal_sidecars_images_and_regrades(tmp_path):
    result_path = tmp_path / "results_Host_20260101_000000_vllm.json"
    names = {path.name for path in run_artifact_paths(result_path, tmp_path)}
    assert names == {
        result_path.name, "results_Host_20260101_000000_vllm.events.sqlite3",
        "results_Host_20260101_000000_vllm.events.sqlite3-wal",
        "results_Host_20260101_000000_vllm.events.sqlite3-shm",
        "results_Host_20260101_000000_vllm.events.sqlite3-journal",
        "log_Host_20260101_000000_vllm.txt",
        "images_Host_20260101_000000_vllm",
        "regraded_results_Host_20260101_000000_vllm.json",
        *(f"answers_{workload}_Host_20260101_000000_vllm.json"
          for workload in ("mcq", "math", "reasoning", "code", "tool")),
        *(f"regraded_answers_{workload}_Host_20260101_000000_vllm.json"
          for workload in ("mcq", "math", "reasoning", "code", "tool")),
    }


def test_delete_run_artifacts_removes_exact_set_and_preserves_neighbors(tmp_path):
    result_path = tmp_path / "results_run.json"
    result_path.write_text("{}", encoding="utf-8")
    journal = result_path.with_suffix(".events.sqlite3")
    journal.write_bytes(b"sqlite")
    wal = Path(f"{journal}-wal")
    wal.write_bytes(b"wal")
    shm = Path(f"{journal}-shm")
    shm.write_bytes(b"shm")
    answer = tmp_path / "answers_mcq_run.json"
    answer.write_text("{}", encoding="utf-8")
    log = tmp_path / "log_run.txt"
    log.write_text("run output", encoding="utf-8")
    images = tmp_path / "images_run"
    images.mkdir()
    (images / "sample.png").write_bytes(b"png")
    regraded = tmp_path / "regraded_results_run.json"
    regraded.write_text("{}", encoding="utf-8")
    unrelated = tmp_path / "answers_mcq_run_extra.json"
    unrelated.write_text("{}", encoding="utf-8")

    removed, failures = delete_run_artifacts(result_path, tmp_path)

    assert failures == {}
    assert set(removed) == {result_path, journal, wal, shm, answer, log, images, regraded}
    assert unrelated.is_file()
    assert existing_run_artifacts(result_path, tmp_path) == []


def test_regraded_selection_does_not_delete_its_source_result(tmp_path):
    source = tmp_path / "results_run.json"
    source.write_text("{}", encoding="utf-8")
    regraded = tmp_path / "regraded_results_run.json"
    regraded.write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "regraded_answers_math_run.json"
    sidecar.write_text("{}", encoding="utf-8")
    images = tmp_path / "images_run"
    images.mkdir()
    log = tmp_path / "log_run.txt"
    log.write_text("source log", encoding="utf-8")

    removed, failures = delete_run_artifacts(regraded, tmp_path)

    assert failures == {}
    assert set(removed) == {regraded, sidecar}
    assert source.is_file()
    assert images.is_dir()
    assert log.is_file()


def test_regraded_custom_name_deletes_its_sidecar_without_touching_source(tmp_path):
    source = tmp_path / "custom.json"
    source.write_text("{}", encoding="utf-8")
    regraded = tmp_path / "regraded_custom.json"
    regraded.write_text("{}", encoding="utf-8")
    sidecar = tmp_path / "regraded_answers_tool_custom.json"
    sidecar.write_text("{}", encoding="utf-8")

    removed, failures = delete_run_artifacts(regraded, tmp_path)

    assert failures == {}
    assert set(removed) == {regraded, sidecar}
    assert source.is_file()


def test_run_artifact_paths_reject_result_outside_history_directory(tmp_path):
    outside = tmp_path.parent / "results_outside.json"
    with pytest.raises(ValueError, match="outside the results directory"):
        run_artifact_paths(outside, tmp_path)


def test_delete_unlinks_image_symlink_without_following_it(tmp_path):
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks unavailable")
    result_path = tmp_path / "results_run.json"
    result_path.write_text("{}", encoding="utf-8")
    target = tmp_path / "kept"
    target.mkdir()
    (target / "image.png").write_bytes(b"png")
    link = tmp_path / "images_run"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    removed, failures = delete_run_artifacts(result_path, tmp_path)

    assert failures == {}
    assert link in removed and not link.exists()
    assert (target / "image.png").is_file()


def test_delete_failure_retains_main_result_for_visible_retry(tmp_path, monkeypatch):
    result_path = tmp_path / "results_run.json"
    result_path.write_text("{}", encoding="utf-8")
    images = tmp_path / "images_run"
    images.mkdir()
    monkeypatch.setattr(
        "scripts.results.result_history.shutil.rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("busy")),
    )

    removed, failures = delete_run_artifacts(result_path, tmp_path)

    assert removed == []
    assert failures == {images: "busy"}
    assert result_path.is_file()


def test_metric_extraction_uses_named_supported_evidence_only():
    metrics = extract_comparable_metrics(result())
    assert metrics["llm/model/2K/tps_mean"]["value"] == 50.0
    assert metrics["embeddings/embed/chunks_per_sec_mean"]["value"] == 100.0
    assert metrics["images/flux/1024x1024/sec_per_image_mean"]["value"] == 8.0
    assert metrics["mcq/model/accuracy_pct"]["value"] == 75.0


def test_comparison_reports_exact_deltas_for_compatible_results():
    comparison = compare_results(result(tps=50.0), result(tps=55.0))
    assert comparison["compatible"] is True
    row = next(item for item in comparison["rows"] if item["metric"] == "llm/model/2K/tps_mean")
    assert row["baseline"] == 50.0
    assert row["candidate"] == 55.0
    assert row["delta"] == 5.0
    assert row["percent_change"] == 10.0
    assert row["within_run_uncertainty"] == "insufficient"
    assert row["verdict"] == "repeated_trials_required"


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
