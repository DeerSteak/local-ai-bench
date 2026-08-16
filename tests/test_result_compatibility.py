import json
from pathlib import Path

import pytest

from scripts.results.result_store import model_counts, validate_json_data


FIXTURES = Path(__file__).parent / "fixtures"
REQUIRED_SECTIONS = {
    "llm", "llm_conversation", "embeddings", "images", "mcq", "math",
    "reasoning", "code", "tool", "concurrency_tool", "concurrency_chat",
    "llamabench", "llamabenchconc",
}


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_schema_1_fixture_preserves_legacy_aggregate_only_result():
    result = load_fixture("results_v4_1_schema1_legacy.json")
    validate_json_data(result)
    assert result["run"]["schema_version"] == 1
    assert result["llm"]["golden"]["2K"] == {
        "ttft_mean_sec": 0.25,
        "tps_mean": 50.0,
        "n_runs": 1,
    }


@pytest.mark.parametrize("name", [
    "results_v4_1_complete.json",
    "results_v4_1_interrupted.json",
])
def test_v4_1_golden_results_remain_valid_finite_json(name):
    result = load_fixture(name)
    validate_json_data(result)
    assert result["version"] == "4.1"
    assert result["run"]["schema_version"] == 2
    assert REQUIRED_SECTIONS <= result.keys()


def test_schema_3_fixture_carries_a_reproducible_plan_identity():
    from scripts.results.run_plan import RunPlan

    result = load_fixture("results_v4_1_schema3_plan.json")
    validate_json_data(result)
    assert result["run"]["schema_version"] == 3
    plan = RunPlan.from_dict(result["run"]["plan"])
    assert plan.plan_id == result["run"]["plan_id"]
    assert plan.models == result["run"]["models"]
    assert plan.effective_config == result["run"]["effective_config"]


def test_schema_4_fixture_preserves_pause_evidence_and_measurements():
    result = load_fixture("results_v4_1_schema4_pause.json")
    validate_json_data(result)
    assert result["run"]["schema_version"] == 4
    assert result["run"]["pause"]["control_transitions"] == [
        {"state": "paused", "timestamp": "2026-01-04T00:03:00+00:00"},
        {"state": "running", "timestamp": "2026-01-04T00:04:30+00:00"},
    ]
    assert result["llm"]["golden"]["2K"]["tps_mean"] == 50.0


def test_schema_5_fixture_retains_memory_and_power_samples_and_run_summaries():
    result = load_fixture("results_v6_schema5_memory.json")
    validate_json_data(result)
    memory = result["llm"]["golden"]["2K"]["memory"]
    assert result["run"]["schema_version"] == 5
    assert [window["name"] for window in memory["windows"]] == [
        "idle", "model_load", "measured",
    ]
    assert memory["windows"][2]["samples"][0]["process_rss_gb"] == 5.0
    assert result["run"]["memory_summary"]["tightest_headroom"]["case_id"] == "golden-memory-case"
    power = result["llm"]["golden"]["2K"]["power"]
    assert power["scope"] == "accelerator"
    assert power["windows"][1]["samples"][-1] == {"timestamp_sec": 3.0, "watts": 14.0}
    assert power["efficiency"] == {
        "unit": "tokens_per_joule", "work_count": 120, "per_joule": 10.0,
    }
    assert result["run"]["power_summary"]["energy_joules"] == 12.0


def test_v4_1_complete_fixture_freezes_coverage_and_measurement_contract():
    result = load_fixture("results_v4_1_complete.json")
    assert result["run"]["status"] == "complete"
    for stage, section in (("llm", "llm"), ("conv", "llm_conversation"),
                           ("llamabench", "llamabench")):
        record = result["run"]["stages"][stage]
        assert record["status"] == "complete"
        assert model_counts(result[section]) == {
            "models_with_results": record["models_with_results"],
            "models_skipped": record["models_skipped"],
            "models_failed": record["models_failed"],
        }
    sample = result["llm"]["golden"]["2K"]["valid_samples"][0]
    assert set(sample) == {
        "client_ttft_sec", "server_prompt_sec", "client_wall_sec", "decode_sec",
        "generated_tokens", "tokens_per_sec", "finish_reason", "model_load_sec",
    }


def test_v4_1_interrupted_fixture_preserves_usable_partial_measurement():
    result = load_fixture("results_v4_1_interrupted.json")
    assert result["run"]["status"] == "interrupted"
    assert result["run"]["stages"]["llm"]["status"] == "interrupted"
    checkpoint = result["llm"]["golden"]["0.5K"]
    assert checkpoint["completed_runs"] == 1
    assert checkpoint["requested_runs"] == 3
    assert checkpoint["valid_runs"] == 1
    assert checkpoint["valid_samples"][0]["tokens_per_sec"] == 50.0
