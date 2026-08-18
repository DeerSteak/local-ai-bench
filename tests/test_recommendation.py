import pytest

from scripts.results.recommendation import evaluate_recommendation, parse_constraints
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
    parsed = parse_constraints({"workload": "llm"})
    assert parsed.maximum_ttft_sec is None
    assert parsed.minimum_throughput is None
    with pytest.raises(ValueError, match="accuracy_section is required"):
        parse_constraints({"workload": "llm", "minimum_accuracy_pct": 0})
    with pytest.raises(ValueError, match="concurrency is only valid"):
        parse_constraints({"workload": "llm", "concurrency": 2})
    with pytest.raises(ValueError, match="at most 100"):
        parse_constraints({"workload": "llm", "accuracy_section": "code",
                           "minimum_accuracy_pct": 101})


def test_hard_filters_run_before_ranking_and_name_the_eliminating_measurement():
    artifact = evaluate_recommendation(result(), request())
    assert artifact["verdict"] == "recommended"
    assert [item["candidate"] for item in artifact["recommended"]] == ["accurate"]
    eliminated = artifact["eliminated"][0]
    assert eliminated["candidate"] == "fast"
    assert eliminated["reasons"] == [{
        "constraint": "accuracy", "operator": "minimum", "threshold": 80.0,
            "measurement": {
                "value": 70.0, "unit": "percent", "evidence_path": "code/fast/accuracy_pct",
                "trial_values": [70.0],
            },
    }]
    assert "partial" not in [item["candidate"] for item in artifact["recommended"]]


def test_missing_case_is_unevaluated_not_eliminated_and_names_the_run_needed():
    artifact = evaluate_recommendation(result(), request())
    partial = next(item for item in artifact["unevaluated"] if item["candidate"] == "partial")
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
    assert [item["candidate"] for item in exact["recommended"]] == ["accurate"]
    failed = evaluate_recommendation(result(), request(
        minimum_accuracy_pct=90.01, maximum_ttft_sec=0.59, minimum_throughput=60.01,
        maximum_memory_gb=11.99, minimum_memory_headroom_gb=12.01,
        minimum_efficiency_per_joule=9.01,
    ))
    reasons = {reason["constraint"] for item in failed["eliminated"]
               if item["candidate"] == "accurate" for reason in item["reasons"]}
    assert reasons == {"accuracy", "ttft", "throughput", "memory", "memory_headroom", "efficiency"}


def test_multiple_survivors_require_repeated_trials_and_emit_no_ranked_list():
    artifact = evaluate_recommendation(result(), request(minimum_accuracy_pct=60))
    assert artifact["verdict"] == "insufficient_evidence"
    assert artifact["recommended"] == []
    assert artifact["tied"] == []
    assert {item["candidate"] for item in artifact["unevaluated"]
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
    assert {item["candidate"] for item in tied["tied"]} == {"fast", "accurate"}
    recommended = evaluate_recommendation(
        repeated_results(
            [90, 90.2, 89.8, 90.1, 89.9],
            [70, 70.2, 69.8, 70.1, 69.9],
        ),
        request(minimum_accuracy_pct=60),
    )
    assert recommended["verdict"] == "recommended"
    assert recommended["recommended"][0]["candidate"] == "fast"


def test_incompatible_repeated_results_are_rejected_not_pooled():
    values = repeated_results([80] * 5, [70] * 5)
    values[-1]["profile"]["hostname"] = "other"
    with pytest.raises(ValueError, match="hardware_identity"):
        evaluate_recommendation(values, request(minimum_accuracy_pct=60))


def test_no_code_path_emits_a_composite_score():
    artifact = evaluate_recommendation(result(), request())
    assert "score" not in repr(artifact).lower()
