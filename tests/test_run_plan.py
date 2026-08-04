from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from run_plan import IDENTITY_SCHEME, PLAN_SCHEMA_VERSION, RunPlan, load_run_plan


def make_plan(**overrides):
    values = {
        "application_version": "4.1",
        "engine_name": "llamacpp",
        "tests": ["llm", "conv"],
        "stage_order": ["llm", "conv"],
        "models": {
            "llm": [{"tag": "model:4b", "short": "model", "size_gb": 3.0}],
            "concurrency": [], "embeddings": [], "images": [],
        },
        "effective_config": {
            "runs": 3, "warmup_runs": 2, "cpu_only": False, "force_all": False,
            "max_prompt_tokens": 32768, "context_lengths": [512, 2048, 8192, 32768],
        },
    }
    values.update(overrides)
    return RunPlan.create(**values)


def test_plan_is_immutable_and_round_trips_through_public_dict():
    plan = make_plan()
    with pytest.raises(FrozenInstanceError):
        plan.engine_name = "other"
    encoded = plan.to_dict()
    assert encoded["schema_version"] == PLAN_SCHEMA_VERSION
    assert RunPlan.from_dict(encoded) == plan
    assert RunPlan.from_dict(encoded).plan_id == plan.plan_id


def test_plan_identity_is_deterministic_across_mapping_order():
    plan = make_plan()
    reordered_config = dict(reversed(list(plan.effective_config.items())))
    reordered_models = dict(reversed(list(plan.models.items())))
    equivalent = make_plan(models=reordered_models, effective_config=reordered_config)
    assert equivalent.plan_id == plan.plan_id
    assert equivalent.job_id != plan.job_id
    assert len(plan.plan_id) == 64


def test_schema_two_derives_stable_hierarchical_execution_ids():
    plan = make_plan()
    restored = RunPlan.from_dict(plan.to_dict())
    identity = plan.models["llm"][0]
    model_id = plan.model_id("llm", identity)
    case_id = plan.case_id("llm", model_id, {"context_tokens": 2048})
    attempt_id = plan.attempt_id(case_id, 1)
    assert plan.to_dict()["identity_scheme"] == IDENTITY_SCHEME
    assert plan.job_id.startswith("job_")
    assert plan.job_id == restored.job_id
    assert plan.stage_id("llm") == restored.stage_id("llm")
    assert model_id == restored.model_id("llm", identity)
    assert case_id == restored.case_id("llm", model_id, {"context_tokens": 2048})
    assert plan.sample_id(attempt_id, 1) == restored.sample_id(attempt_id, 1)
    assert len({plan.job_id, plan.stage_id("llm"), model_id, case_id, attempt_id,
                plan.sample_id(attempt_id, 1)}) == 6


def test_execution_ids_reject_entities_outside_plan_and_invalid_ordinals():
    plan = make_plan()
    with pytest.raises(ValueError, match="stage is not present"):
        plan.stage_id("img")
    with pytest.raises(ValueError, match="model identity is not present"):
        plan.model_id("llm", {"tag": "missing"})
    with pytest.raises(ValueError, match="positive integer"):
        plan.attempt_id("case", 0)
    with pytest.raises(ValueError, match="positive integer"):
        plan.sample_id("attempt", True)


@pytest.mark.parametrize("change", ["engine", "tests", "models", "config"])
def test_measurement_affecting_changes_produce_a_new_plan_identity(change):
    base = make_plan()
    if change == "engine":
        changed = make_plan(engine_name="other")
    elif change == "tests":
        changed = make_plan(tests=["llm"], stage_order=["llm"])
    elif change == "models":
        models = base.models
        models["llm"] = [{"tag": "other:8b", "short": "other"}]
        changed = make_plan(models=models)
    else:
        config = base.effective_config
        config["runs"] = 4
        changed = make_plan(effective_config=config)
    assert changed.plan_id != base.plan_id


def test_returned_models_and_config_cannot_mutate_the_plan():
    plan = make_plan()
    plan.models["llm"].append({"tag": "injected"})
    plan.effective_config["runs"] = 99
    assert plan.models["llm"] == [{"short": "model", "size_gb": 3.0, "tag": "model:4b"}]
    assert plan.effective_config["runs"] == 3


@pytest.mark.parametrize("field", ["output_path", "hf_token", "model_path"])
def test_plan_rejects_unknown_or_secret_bearing_settings(field):
    config = make_plan().effective_config
    config[field] = "/private/value"
    with pytest.raises(ValueError, match="unsafe or unknown"):
        make_plan(effective_config=config)


def test_plan_rejects_paths_or_secrets_in_model_identity():
    models = make_plan().models
    models["llm"][0]["path"] = "/private/model.gguf"
    with pytest.raises(ValueError, match="unsafe or unknown model identity"):
        make_plan(models=models)


def test_plan_rejects_mismatched_or_duplicate_stage_sets():
    with pytest.raises(ValueError, match="same stages"):
        make_plan(stage_order=["llm"])
    with pytest.raises(ValueError, match="duplicates"):
        make_plan(tests=["llm", "llm"], stage_order=["llm", "llm"])


def test_plan_rejects_nonfinite_configuration():
    config = make_plan().effective_config
    config["run_timeout_seconds"] = float("nan")
    with pytest.raises(ValueError):
        make_plan(effective_config=config)


def test_plan_rejects_unsupported_schema():
    encoded = make_plan().to_dict()
    encoded["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported"):
        RunPlan.from_dict(encoded)


def test_schema_one_plan_preserves_legacy_identity_and_round_trip():
    fixture = Path(__file__).parent / "fixtures" / "results_v4_1_schema3_plan.json"
    result = json.loads(fixture.read_text(encoding="utf-8"))
    legacy = RunPlan.from_dict(result["run"]["plan"])
    assert legacy.schema_version == 1
    assert legacy.to_dict() == result["run"]["plan"]
    assert legacy.plan_id == result["run"]["plan_id"]


def test_schema_two_requires_known_identity_scheme():
    encoded = make_plan().to_dict()
    encoded["identity_scheme"] = "other"
    with pytest.raises(ValueError, match="identity scheme"):
        RunPlan.from_dict(encoded)


def test_schema_two_requires_serialized_job_identity():
    encoded = make_plan().to_dict()
    encoded.pop("job_id")
    with pytest.raises(ValueError, match="job identity"):
        RunPlan.from_dict(encoded)


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_run_plan_accepts_plan_or_cli_results(tmp_path, wrapped):
    plan = make_plan()
    encoded = plan.to_dict()
    document = {"run": {"plan": encoded}} if wrapped else encoded
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_run_plan(path) == plan


def test_load_run_plan_rejects_unrelated_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises((KeyError, ValueError)):
        load_run_plan(path)


def test_load_run_plan_rejects_malformed_result_wrapper(tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"run": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain"):
        load_run_plan(path)
