import json

from scripts.app.benchmark import format_dry_run_output, format_duration_estimate, format_resolved_plan
from scripts.results.result_history import (
    completed_run_duration_seconds, estimate_matching_plan_seconds,
)


def test_completed_duration_requires_a_positive_completed_interval():
    result = {"run": {"status": "complete", "started_at": "2026-01-01T10:00:00Z",
                      "finished_at": "2026-01-01T10:02:30Z"}}
    assert completed_run_duration_seconds(result) == 150
    result["run"]["status"] = "interrupted"
    assert completed_run_duration_seconds(result) is None
    result["run"].update(status="complete", started_at="2026-01-01T10:00:00",
                         finished_at="2026-01-01T10:02:30Z")
    assert completed_run_duration_seconds(result) is None


def test_eta_uses_only_exact_completed_plan_matches(tmp_path):
    models = {"llm": [{"short": "tiny"}], "concurrency": [], "embeddings": [], "images": []}
    for name, seconds, engine in (("a", 120, "llamacpp"), ("b", 240, "llamacpp"), ("c", 9, "vllm")):
        result = {
            "engine": engine, "profile": {"hostname": "host"},
            "run": {
                "status": "complete", "started_at": "2026-01-01T10:00:00+00:00",
                "finished_at": f"2026-01-01T10:{seconds // 60:02d}:{seconds % 60:02d}+00:00",
                "plan": {"requested_tests": ["llm"], "models": models},
            },
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(result), encoding="utf-8")
    assert estimate_matching_plan_seconds(tmp_path, "llamacpp", ["llm"], models) == 180


def test_eta_treats_workload_order_as_plan_equivalent(tmp_path):
    models = {"llm": [{"short": "tiny"}], "concurrency": [], "embeddings": [], "images": []}
    result = {
        "engine": "llamacpp",
        "run": {
            "status": "complete", "started_at": "2026-01-01T10:00:00+00:00",
            "finished_at": "2026-01-01T10:01:00+00:00",
            "plan": {"requested_tests": ["conv", "llm"], "models": models},
        },
    }
    (tmp_path / "ordered.json").write_text(json.dumps(result), encoding="utf-8")
    assert estimate_matching_plan_seconds(tmp_path, "llamacpp", ["llm", "conv"], models) == 60


def test_dry_run_output_explains_an_empty_engine_selection():
    assert format_dry_run_output([]) == "No workloads resolved for the selected engine pass(es)."
    assert format_dry_run_output(["first", "second"]) == "first\n\nsecond"


def test_resolved_plan_lists_models_cases_and_historical_eta(monkeypatch):
    from scripts.runtime import config
    monkeypatch.setattr(config, "CONTEXT_LENGTHS", [512, 2048])
    monkeypatch.setattr(config, "N_RUNS", 1)
    monkeypatch.setattr(config, "WARMUP_RUNS", 0)
    models = {"llm": [{"short": "tiny", "label": "Tiny"}], "concurrency": [],
              "embeddings": [], "images": []}
    preview = format_resolved_plan(
        "llamacpp", ["llm"], models, 150, runs=1, warmups=0,
        max_prompt_tokens=2048, sample_size=None,
    )
    assert "llm: Tiny — contexts 512, 2048" in preview
    assert "Runs: 1 measured + 0 warmup" in preview
    assert "Estimated duration: about 2m" in preview
    assert "no exact completed local plan match" in format_duration_estimate(None)
