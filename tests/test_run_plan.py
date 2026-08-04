from dataclasses import FrozenInstanceError
import json

import pytest

from run_plan import PLAN_SCHEMA_VERSION, RunPlan, load_run_plan


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
    assert len(plan.plan_id) == 64


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


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_run_plan_accepts_plan_or_cli_results(tmp_path, wrapped):
    encoded = make_plan().to_dict()
    document = {"run": {"plan": encoded}} if wrapped else encoded
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_run_plan(path) == make_plan()


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
