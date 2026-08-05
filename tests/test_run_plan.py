from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from scripts.results.run_plan import IDENTITY_SCHEME, PLAN_SCHEMA_VERSION, RunPlan, load_run_plan


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


def test_current_schema_derives_stable_hierarchical_execution_ids():
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


def test_methodology_profile_is_identity_bearing_and_validated():
    config = complete_plan().effective_config
    config.update({
        "methodology_profile": "neutral-v1",
        "effective_optimizations": ["llamacpp:flash_attention=on"],
    })
    plan = make_plan(effective_config=config)
    methodology = plan.execution_identity["methodology"]
    assert methodology == {
        "profile": "neutral-v1",
        "effective_optimizations": ["llamacpp:flash_attention=on"],
    }
    changed = dict(config, effective_optimizations=["llamacpp:flash_attention=off"])
    assert make_plan(effective_config=changed).plan_id != plan.plan_id
    invalid = dict(config, methodology_profile="vendor-fast")
    with pytest.raises(ValueError, match="methodology_profile"):
        make_plan(effective_config=invalid).validate_for_execution()


def test_offline_mode_is_identity_bearing_without_changing_legacy_plans():
    legacy = make_plan()
    config = complete_plan().effective_config
    config["offline"] = True
    offline = make_plan(effective_config=config)
    assert offline.execution_identity["privacy"]["offline"] is True
    assert "offline" not in legacy.execution_identity["privacy"]
    assert offline.plan_id != make_plan(effective_config=dict(config, offline=False)).plan_id


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


def test_current_schema_requires_known_identity_scheme():
    encoded = make_plan().to_dict()
    encoded["identity_scheme"] = "other"
    with pytest.raises(ValueError, match="identity scheme"):
        RunPlan.from_dict(encoded)


def test_current_schema_requires_serialized_job_identity():
    encoded = make_plan().to_dict()
    encoded.pop("job_id")
    with pytest.raises(ValueError, match="job identity"):
        RunPlan.from_dict(encoded)


def test_current_schema_covers_execution_policy_identities_without_paths():
    plan = complete_plan()
    identity = plan.to_dict()["execution_identity"]
    assert identity["workloads"] == {"llm": "4.1", "conv": "4.1"}
    assert identity["runtime"] == {"engine": "llamacpp", "adapter_contract": 1}
    assert identity["retry"]["implausible_tps_retries"] == 1
    assert identity["timeouts"] == {"run_seconds": 1800, "accuracy_seconds": 60}
    assert identity["output"] == {"result_schema": 3, "event_schema": 1}
    assert "/" not in json.dumps(identity)


def test_current_schema_rejects_changed_or_missing_execution_identity():
    encoded = complete_plan().to_dict()
    encoded["execution_identity"]["retry"]["implausible_tps_retries"] = 2
    with pytest.raises(ValueError, match="execution identity"):
        RunPlan.from_dict(encoded)
    encoded = complete_plan().to_dict()
    encoded.pop("execution_identity")
    with pytest.raises(ValueError, match="execution identity"):
        RunPlan.from_dict(encoded)


def test_schema_two_remains_readable_without_execution_identity():
    plan = make_plan(schema_version=2)
    encoded = plan.to_dict()
    assert "execution_identity" not in encoded
    assert RunPlan.from_dict(encoded) == plan


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


def complete_plan():
    config = make_plan().effective_config
    config.update({
        "run_timeout_seconds": 1800, "accuracy_timeout_seconds": 60,
        "accuracy_token_budget": 4096, "llamabench_pp": [512, 2048],
        "llamabench_tg": [128, 256], "sample_size": None,
        "concurrency_tool_levels": [1, 2], "concurrency_chat_levels": [1, 2, 4],
        "concurrency_tool_context": 4096, "concurrency_chat_context": 16384,
        "concurrency_chat_soft_exit_floor": 8,
    })
    return make_plan(effective_config=config)


def test_complete_plan_validation_accepts_resolved_execution_inputs():
    complete_plan().validate_for_execution()


@pytest.mark.parametrize(("key", "value"), [
    ("runs", 0), ("warmup_runs", -1), ("cpu_only", 1),
    ("context_lengths", [512, 512]), ("llamabench_pp", []),
    ("llamabench_tg", [True]), ("sample_size", 0),
    ("concurrency_tool_levels", [1, 1]), ("concurrency_chat_context", 0),
])
def test_complete_plan_validation_rejects_invalid_resolved_settings(key, value):
    plan = complete_plan()
    config = plan.effective_config
    config[key] = value
    invalid = make_plan(effective_config=config)
    with pytest.raises(ValueError, match="invalid execution setting"):
        invalid.validate_for_execution()


def test_complete_plan_validation_rejects_missing_settings_and_model_identity():
    with pytest.raises(ValueError, match="incomplete run plan"):
        make_plan().validate_for_execution()
    plan = complete_plan()
    models = plan.models
    models["llm"] = [{"tag": "model:4b", "short": ""}]
    invalid = make_plan(models=models, effective_config=plan.effective_config)
    with pytest.raises(ValueError, match="invalid model identity"):
        invalid.validate_for_execution()


def test_complete_plan_validation_accepts_short_only_image_identity():
    plan = complete_plan()
    models = plan.models
    models["images"] = [{"short": "sdxl"}]
    make_plan(models=models, effective_config=plan.effective_config).validate_for_execution()


def test_complete_plan_validation_rejects_image_without_short_identity():
    plan = complete_plan()
    models = plan.models
    models["images"] = [{"tag": "sdxl"}]
    invalid = make_plan(models=models, effective_config=plan.effective_config)
    with pytest.raises(ValueError, match="invalid model identity in family: images"):
        invalid.validate_for_execution()
