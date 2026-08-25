import json
from pathlib import Path

import pytest

from scripts.results.recommendation import (
    candidate_evidence, evaluate_recommendation, parse_constraints,
    validate_recommendation_artifact,
)
from scripts.results.recommendation_cli import main
from tests.test_result_history import result as history_result


def result():
    def case(tps, ttft, memory, headroom, efficiency):
        return {
            "tps_mean": tps, "ttft_mean_sec": ttft,
            "memory": {
                "summary": {"process_rss_gb": {"peak_gb": memory}},
                "headroom": {"absolute_gb": headroom},
            },
            "power": {"efficiency": {"per_joule": efficiency, "unit": "tokens_per_joule"}},
        }
    return {
        "version": "6.0", "engine": "llamacpp", "profile": {"hostname": "system"},
        "run": {"status": "complete", "plan": {"effective_config": {
            "methodology_profile": "neutral-v1",
        }}},
        "llm": {
            "fast": {"8K": case(80, 0.4, 8, 16, 12)},
            "accurate": {"8K": case(60, 0.6, 12, 12, 9)},
            "partial": {"2K": case(100, 0.2, 7, 17, 13)},
        },
        "code": {
            "fast": {"accuracy_pct": 70},
            "accurate": {"accuracy_pct": 90},
            "partial": {"accuracy_pct": 95},
        },
    }


def request(**changes):
    value = {
        "workload": "llm", "case": "8K", "accuracy_section": "code",
        "primary_objective": "throughput", "minimum_accuracy_pct": 80,
    }
    value.update(changes)
    return value


def test_constraints_preserve_absent_values_and_reject_invalid_combinations():
    parsed = parse_constraints({"workload": "llm", "case": "8K"})
    assert parsed.maximum_ttft_sec is None
    assert parsed.minimum_throughput is None
    with pytest.raises(ValueError, match="accuracy_section is required"):
        parse_constraints({"workload": "llm", "minimum_accuracy_pct": 0})
    with pytest.raises(ValueError, match="concurrency is only valid"):
        parse_constraints({"workload": "llm", "concurrency": 2})
    with pytest.raises(ValueError, match="at most 100"):
        parse_constraints({"workload": "llm", "accuracy_section": "code",
                           "minimum_accuracy_pct": 101})
    with pytest.raises(ValueError, match="unknown recommendation constraint fields"):
        parse_constraints({"workload": "llm", "score_weights": {"speed": 1}})
    with pytest.raises(ValueError, match="concurrency or case"):
        parse_constraints({"workload": "concurrency_chat", "case": "2", "concurrency": 2})
    with pytest.raises(ValueError, match="not available for image"):
        parse_constraints({"workload": "images", "case": "1024x1024",
                           "accuracy_section": "code"})
    with pytest.raises(ValueError, match="case is required"):
        parse_constraints({"workload": "llm"})


def test_hard_filters_run_before_ranking_and_name_the_eliminating_measurement():
    artifact = evaluate_recommendation(result(), request())
    assert artifact["verdict"] == "recommended"
    assert [item["candidate"] for item in artifact["candidates"]["recommended"]] == ["accurate"]
    eliminated = artifact["candidates"]["eliminated"][0]
    assert eliminated["candidate"] == "fast"
    assert eliminated["reasons"] == [{
        "constraint": "accuracy", "operator": "minimum", "threshold": 80.0,
            "measurement": {
                "value": 70.0, "unit": "percent", "evidence_path": "code/fast/accuracy_pct",
                "raw_evidence_paths": ["answers_code/fast"],
                "trial_values": [70.0],
            },
    }]


def test_quantization_variants_enter_the_existing_constraint_first_candidate_pool():
    data = result()
    data["llm"] = {
        "demo-q4": data["llm"]["fast"],
        "demo-q8": data["llm"]["accurate"],
    }
    data["code"] = {
        "demo-q4": {"accuracy_pct": 70},
        "demo-q8": {"accuracy_pct": 90},
    }
    data["run"]["plan"]["models"] = {"llm": [
        {"short": "demo-q4", "base_model": "demo", "variant": "Q4_K_M"},
        {"short": "demo-q8", "base_model": "demo", "variant": "Q8_0"},
    ]}

    artifact = evaluate_recommendation(data, request())

    assert [item["candidate"] for item in artifact["candidates"]["recommended"]] == [
        "demo-q8",
    ]
    assert [item["candidate"] for item in artifact["candidates"]["eliminated"]] == [
        "demo-q4",
    ]
    assert "partial" not in [item["candidate"] for item in artifact["candidates"]["recommended"]]


def test_missing_case_is_unevaluated_not_eliminated_and_names_the_run_needed():
    artifact = evaluate_recommendation(result(), request())
    partial = next(item for item in artifact["candidates"]["unevaluated"]
                   if item["candidate"] == "partial")
    assert partial["missing_evidence"] == ["throughput"]
    assert partial["resolution"] == {
        "workload": "llm", "case": "8K", "accuracy_section": "code",
    }


def test_every_constraint_type_filters_at_its_boundary():
    exact = evaluate_recommendation(result(), request(
        minimum_accuracy_pct=90, maximum_ttft_sec=0.6, minimum_throughput=60,
        maximum_memory_gb=12, minimum_memory_headroom_gb=12,
        minimum_efficiency_per_joule=9,
    ))
    assert [item["candidate"] for item in exact["candidates"]["recommended"]] == ["accurate"]
    failed = evaluate_recommendation(result(), request(
        minimum_accuracy_pct=90.01, maximum_ttft_sec=0.59, minimum_throughput=60.01,
        maximum_memory_gb=11.99, minimum_memory_headroom_gb=12.01,
        minimum_efficiency_per_joule=9.01,
    ))
    reasons = {reason["constraint"] for item in failed["candidates"]["eliminated"]
               if item["candidate"] == "accurate" for reason in item["reasons"]}
    assert reasons == {"accuracy", "ttft", "throughput", "memory", "memory_headroom", "efficiency"}


def test_multiple_survivors_require_repeated_trials_and_emit_no_ranked_list():
    artifact = evaluate_recommendation(result(), request(minimum_accuracy_pct=60))
    assert artifact["verdict"] == "insufficient_evidence"
    assert artifact["candidates"]["recommended"] == []
    assert artifact["candidates"]["tied"] == []
    assert artifact["candidates"]["other_eligible"] == []
    assert {item["candidate"] for item in artifact["candidates"]["unevaluated"]
            if item["missing_evidence"] == ["qualified_repeated_trial_verdict"]} == {
                "accurate", "fast",
            }


def repeated_results(fast_values, accurate_values):
    results = []
    for index, (fast, accurate) in enumerate(zip(fast_values, accurate_values)):
        value = result()
        value.update({
            "version": "4.1", "engine": "llamacpp", "profile": {"hostname": "system"},
            "run": history_result(started=f"2026-01-{index + 1:02d}T00:00:00Z")["run"],
        })
        value["llm"]["fast"]["8K"]["tps_mean"] = fast
        value["llm"]["accurate"]["8K"]["tps_mean"] = accurate
        results.append(value)
    return results


def test_qualified_repeated_trials_produce_tied_or_recommended_verdicts():
    tied = evaluate_recommendation(
        repeated_results(
            [80, 80.2, 79.8, 80.1, 79.9],
            [79.5, 79.7, 79.3, 79.6, 79.4],
        ),
        request(minimum_accuracy_pct=60),
    )
    assert tied["verdict"] == "tied"
    assert {item["candidate"] for item in tied["candidates"]["tied"]} == {"fast", "accurate"}
    recommended = evaluate_recommendation(
        repeated_results(
            [90, 90.2, 89.8, 90.1, 89.9],
            [70, 70.2, 69.8, 70.1, 69.9],
        ),
        request(minimum_accuracy_pct=60),
    )
    assert recommended["verdict"] == "recommended"
    assert recommended["candidates"]["recommended"][0]["candidate"] == "fast"
    assert [item["candidate"] for item in recommended["candidates"]["other_eligible"]] == [
        "accurate",
    ]


def test_top_tie_is_preserved_when_another_survivor_is_reproducibly_worse():
    results = repeated_results(
        [90, 90.2, 89.8, 90.1, 89.9],
        [89.5, 89.7, 89.3, 89.6, 89.4],
    )
    for value, slow in zip(results, [60, 60.2, 59.8, 60.1, 59.9]):
        value["llm"]["slow"] = {"8K": value["llm"]["accurate"]["8K"] | {
            "tps_mean": slow,
        }}
        value["code"]["slow"] = {"accuracy_pct": 85}
    artifact = evaluate_recommendation(results, request(minimum_accuracy_pct=60))
    assert artifact["verdict"] == "tied"
    assert {item["candidate"] for item in artifact["candidates"]["tied"]} == {
        "fast", "accurate",
    }
    assert [item["candidate"] for item in artifact["candidates"]["other_eligible"]] == [
        "slow",
    ]


def test_incomplete_or_methodology_unknown_evidence_is_unevaluated():
    incomplete = result()
    incomplete["run"]["status"] = "interrupted"
    del incomplete["run"]["plan"]["effective_config"]["methodology_profile"]
    artifact = evaluate_recommendation(incomplete, request())
    assert artifact["verdict"] == "insufficient_evidence"
    accurate = next(item for item in artifact["candidates"]["unevaluated"]
                    if item["candidate"] == "accurate")
    assert accurate["missing_evidence"] == ["complete_run", "methodology_profile"]


def test_incompatible_repeated_results_are_rejected_not_pooled():
    values = repeated_results([80] * 5, [70] * 5)
    values[-1]["profile"]["hostname"] = "other"
    with pytest.raises(ValueError, match="hardware_identity"):
        evaluate_recommendation(values, request(minimum_accuracy_pct=60))


def test_duplicate_results_cannot_satisfy_independent_trial_minimum():
    value = result()
    with pytest.raises(ValueError, match="distinct independent result files"):
        evaluate_recommendation([value] * 5, request(minimum_accuracy_pct=60))


def test_unknown_case_is_a_request_error_with_available_cases():
    with pytest.raises(ValueError, match="unknown case '8k'.*2K, 8K"):
        evaluate_recommendation(result(), request(case="8k"))


def test_unknown_case_does_not_advertise_model_status_markers():
    value = result()
    value["llm"]["fast"].update({
        "crashed": "2K", "crashed_at": "startup", "timed_out": True, "slow_tps": 1,
    })
    with pytest.raises(ValueError, match="unknown case '8k'.*2K, 8K") as error:
        evaluate_recommendation(value, request(case="8k"))
    assert "crashed" not in str(error.value)
    assert "timed_out" not in str(error.value)


def test_status_only_models_remain_unevaluated_for_the_requested_case():
    value = result()
    value["llm"] = {
        "failed": {"crashed": "2K", "crashed_at": "startup"},
    }
    value["code"] = {"failed": {"accuracy_pct": 90}}
    artifact = evaluate_recommendation(value, request())
    assert artifact["verdict"] == "insufficient_evidence"
    assert artifact["candidates"]["unevaluated"] == [{
        "candidate": "failed",
        "missing_evidence": ["throughput"],
        "resolution": {
            "workload": "llm", "case": "8K", "accuracy_section": "code",
        },
    }]
    assert artifact["resolution"] == {
        "action": "run_missing_evidence", "candidate_gaps": 1,
    }


def test_no_code_path_emits_a_composite_score():
    artifact = evaluate_recommendation(result(), request())
    assert "score" not in repr(artifact).lower()


def test_image_throughput_uses_images_per_second_without_reversing_the_filter():
    value = result()
    value["images"] = {
        "quick": {"resolutions": {"1024x1024": {
            "sec_per_image_mean": 3.33, "runs": [3.3, 3.36],
        }}},
        "slow": {"resolutions": {"1024x1024": {
            "sec_per_image_mean": 4, "runs": [3.9, 4.1],
        }}},
    }
    artifact = evaluate_recommendation(value, {
        "workload": "images", "case": "1024x1024", "primary_objective": "throughput",
        "minimum_throughput": 0.29,
    })
    assert artifact["candidates"]["recommended"][0]["candidate"] == "quick"
    evidence = artifact["candidates"]["recommended"][0]["evidence"]["throughput"]
    assert evidence == {
        "value": 0.3003,
        "unit": "images_per_second",
        "evidence_path": "images/quick/resolutions/1024x1024/sec_per_image_mean",
        "raw_evidence_paths": ["images/quick/resolutions/1024x1024/runs"],
        "source_value": 3.33,
        "source_trial_values": [3.33],
        "source_unit": "seconds_per_image",
        "derivation": "reciprocal",
        "trial_values": [0.3003],
    }
    assert artifact["candidates"]["eliminated"][0]["candidate"] == "slow"


def test_empty_candidate_set_and_malformed_verdict_groups_are_explicit():
    value = result()
    value["llm"] = {}
    value["code"] = {}
    artifact = evaluate_recommendation(value, request())
    assert artifact["verdict"] == "insufficient_evidence"
    assert artifact["resolution"] == {
        "action": "run_missing_evidence", "workload": "llm", "case": "8K",
    }
    malformed = evaluate_recommendation(result(), request())
    malformed["candidates"]["recommended"] = []
    with pytest.raises(ValueError, match="does not match"):
        validate_recommendation_artifact(malformed)
    duplicate = evaluate_recommendation(result(), request())
    duplicate["candidates"]["other_eligible"] = list(
        duplicate["candidates"]["recommended"])
    with pytest.raises(ValueError, match="exactly one outcome"):
        validate_recommendation_artifact(duplicate)


def test_shipped_image_sample_uses_real_resolution_paths_and_omits_absent_raw_runs():
    value = json.loads(Path("samples/results_rtx4090-workstation.json").read_text(encoding="utf-8"))
    evidence = candidate_evidence(
        value, parse_constraints({
            "workload": "images", "case": "1024x1024",
            "primary_objective": "throughput",
        }), "sdxl",
    )["throughput"]
    assert evidence["value"] == 0.3521
    assert evidence["evidence_path"] == \
        "images/sdxl/resolutions/1024x1024/sec_per_image_mean"
    assert evidence["raw_evidence_paths"] == []


def test_cli_writes_a_versioned_artifact_and_normalized_constraints(tmp_path):
    result_path = tmp_path / "result.json"
    constraints_path = tmp_path / "constraints.json"
    output_path = tmp_path / "recommendation.json"
    result_path.write_text(json.dumps(result()), encoding="utf-8")
    constraints_path.write_text(json.dumps(request()), encoding="utf-8")
    assert main([
        str(result_path), "--constraints", str(constraints_path), "--out", str(output_path),
    ]) == 0
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "recommendation"
    assert artifact["constraints"]["maximum_ttft_sec"] is None
    assert artifact["verdict"] == "recommended"


def test_cli_rejects_nonfinite_and_malformed_constraint_files(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result()), encoding="utf-8")
    for name, contents in (("nonfinite", '{"workload":"llm","maximum_ttft_sec":NaN}'),
                           ("malformed", "{")):
        constraints_path = tmp_path / f"{name}.json"
        constraints_path.write_text(contents, encoding="utf-8")
        assert main([
            str(result_path), "--constraints", str(constraints_path),
            "--out", str(tmp_path / f"{name}-out.json"),
        ]) == 1


def test_shared_dashboard_conformance_artifact_is_valid():
    artifact = json.loads(Path("samples/recommendation_example.json").read_text(encoding="utf-8"))
    validate_recommendation_artifact(artifact)
    assert artifact["candidates"]["recommended"][0]["candidate"] == "qwen3.5-4b-q4"
