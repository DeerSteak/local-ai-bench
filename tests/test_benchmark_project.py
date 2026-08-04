import copy

import pytest

from scripts.app.benchmark_frontend import GUI_OPTION_DEFAULTS
from scripts.app.benchmark_project import (
    PROJECT_WORKFLOWS, build_project, load_project, project_frontend_state,
    save_project, validate_project,
)


def state():
    return {
        "engine": "llamacpp", "tests": ["llm"],
        "models": {"llm": ["model:4b"], "embedding": [], "image": []},
        "max_prompt_tokens": 8192, "tg_tokens": [128],
        "gui_options": dict(GUI_OPTION_DEFAULTS, out="/private/result.json", comfyui="/private/ComfyUI"),
    }


def policy():
    return {
        "schema_version": 1, "name": "Gate", "methodology_profile": "neutral-v1",
        "rules": [{
            "id": "tps", "section": "llm", "model": "model", "case": "2K",
            "metric": "tps_mean", "operator": "at_least", "threshold": 20.0,
            "minimum_evidence": 3,
        }],
    }


@pytest.mark.parametrize("workflow", PROJECT_WORKFLOWS)
def test_each_decision_workflow_builds_a_valid_local_project(workflow):
    project = build_project(
        "Launch review", workflow, state(), baseline_result="results/baseline.json",
        acceptance_policy=policy(),
    )
    assert project["workflow"] == workflow
    assert project["preset"]["configuration"]["options"] == {
        key: GUI_OPTION_DEFAULTS[key]
            for key in ("warmup", "runs", "timeout", "acc_timeout", "acc_token_budget", "cpu_only", "force_all", "offline")
    }
    encoded = str(project)
    assert "/private/result.json" not in encoded and "/private/ComfyUI" not in encoded


def test_project_round_trip_preserves_embedded_policy_and_baseline(tmp_path):
    project = build_project("Regression", "regression", state(), acceptance_policy=policy())
    path = tmp_path / "project.labproject"
    save_project(path, project)
    assert load_project(path) == project


def test_project_applies_portable_configuration_but_retains_machine_local_paths():
    project = build_project("Selection", "model_selection", state())
    local = dict(GUI_OPTION_DEFAULTS, out="local.json", comfyui="local-comfy")
    restored = project_frontend_state(project, local)
    assert restored["models"]["llm"] == ["model:4b"]
    assert restored["gui_options"]["out"] == "local.json"
    assert restored["gui_options"]["comfyui"] == "local-comfy"


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(schema_version=2),
    lambda value: value.update(workflow="unknown"),
    lambda value: value.update(baseline_result=""),
    lambda value: value.update(extra=True),
    lambda value: value["acceptance_policy"].update(rules=[]),
])
def test_project_rejects_unknown_malformed_or_incomplete_content(mutation):
    project = build_project("Launch", "acceptance_validation", state(), acceptance_policy=policy())
    candidate = copy.deepcopy(project)
    mutation(candidate)
    with pytest.raises(ValueError):
        validate_project(candidate)
