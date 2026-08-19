import json
from pathlib import Path

import pytest

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.release.qualification_automation import (
    execution_recipe_gaps, initial_run_state, load_qualification_recipe, next_qualification_step,
    log_contains_marker, qualification_entry_from_run, qualification_preview, recipe_digest,
    validate_qualification_recipe,
)


def recipe():
    steps = {
        name: {
            "command": ["tool", name], "timeout_seconds": 60,
            "expected_exit_codes": [0], "interrupt_when_log_contains": None,
        }
        for name in QUALIFICATION_LIFECYCLE
    }
    steps["cancellation"].update({
        "expected_exit_codes": [130],
        "interrupt_when_log_contains": '"kind":"model","stage":"llm","status":"running"',
    })
    return {
        "target": {
            "id": "macos-arm64-llamacpp-metal", "platform": "macos",
            "architecture": "arm64", "runtime": "llamacpp",
            "runtime_version": "b6000", "backend": "metal",
            "accelerator": "MacBook Pro / M5 Pro",
        },
        "coverage": {
            "workloads": ["llm"], "models": ["llama3.2:3b-instruct-q4_K_M"],
            "notes": "Lifecycle smoke; not a full catalog performance qualification.",
        },
        "environment": {"HF_HOME": "/qualification/vllm-cache"},
        "steps": steps,
    }


def test_recipe_requires_every_lifecycle_step_and_never_accepts_shell_text():
    value = recipe()
    del value["steps"]["rollback"]
    with pytest.raises(ValueError, match="every lifecycle"):
        validate_qualification_recipe(value)
    value = recipe()
    value["steps"]["install"]["command"] = "dangerous shell command"
    with pytest.raises(ValueError, match="argv command"):
        validate_qualification_recipe(value)


def test_only_cancellation_can_request_an_automatic_interrupt():
    value = recipe()
    value["steps"]["resume"]["interrupt_when_log_contains"] = "running"
    with pytest.raises(ValueError, match="only the cancellation"):
        validate_qualification_recipe(value)


def test_recipe_rejects_serialized_credentials():
    value = recipe()
    value["environment"]["HF_TOKEN"] = "must-not-be-recorded"
    with pytest.raises(ValueError, match="unsafe or unknown"):
        validate_qualification_recipe(value)


def test_recipe_load_and_preview_are_read_only(tmp_path):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe()))
    loaded = load_qualification_recipe(path)
    preview = qualification_preview(loaded, tmp_path / "evidence")
    assert preview["mode"] == "preview"
    assert preview["coverage"]["workloads"] == ["llm"]
    assert preview["checkpoint"].endswith("evidence/qualification-state.json")
    assert [step["name"] for step in preview["steps"]] == list(QUALIFICATION_LIFECYCLE)
    assert not (tmp_path / "evidence").exists()


def test_published_example_recipe_stays_valid():
    root = Path(__file__).resolve().parents[1]
    loaded = load_qualification_recipe(root / "samples/qualification_recipe_example.json")
    assert loaded["target"]["platform"] == "macos"
    assert execution_recipe_gaps(loaded) == [
        "target.runtime_version", "target.accelerator", "coverage.models", "steps.install.command",
        "steps.discovery.command", "steps.first_valid_run.command",
        "steps.cancellation.command", "steps.resume.command",
        "steps.report_generation.command", "steps.bundle_export.command",
        "steps.upgrade.command", "steps.rollback.command", "steps.uninstall.command",
    ]


def test_execution_preflight_accepts_a_fully_resolved_recipe():
    assert execution_recipe_gaps(recipe()) == []


def test_interruption_marker_is_read_from_the_live_step_log(tmp_path):
    log = tmp_path / "step.log"
    assert log_contains_marker(log, '"status":"running"') is False
    log.write_text('prefix {"status":"running"}\n')
    assert log_contains_marker(log, '"status":"running"') is True


def test_checkpoint_resumes_at_first_step_that_has_not_passed():
    state = initial_run_state(recipe())
    assert next_qualification_step(state) == "install"
    state["steps"]["install"]["status"] = "passed"
    state["steps"]["discovery"]["status"] = "failed"
    assert next_qualification_step(state) == "discovery"
    for step in QUALIFICATION_LIFECYCLE:
        state["steps"][step]["status"] = "passed"
    assert next_qualification_step(state) is None


def test_recipe_digest_is_stable_and_changes_with_coverage():
    first = recipe()
    second = recipe()
    assert recipe_digest(first) == recipe_digest(second)
    second["coverage"]["models"] = ["another-model"]
    assert recipe_digest(first) != recipe_digest(second)


def test_run_projects_complete_and_incomplete_steps_into_evidence():
    state = initial_run_state(recipe())
    state["steps"]["install"].update({"status": "passed", "detail": "ok"})
    state["steps"]["discovery"].update({"status": "failed", "detail": "runtime absent"})
    entry = qualification_entry_from_run(state, "6.0-pre8", "records/run.json")
    assert entry["lifecycle"]["install"] == "passed"
    assert entry["lifecycle"]["discovery"] == "failed"
    assert entry["known_failures"][0] == {
        "step": "discovery", "detail": "runtime absent",
    }
    assert entry["evidence"] == ["records/run.json"]
    assert entry["coverage"] == state["coverage"]


@pytest.mark.parametrize("value", [0, -1, True])
def test_timeouts_must_be_positive_integer_seconds(value):
    candidate = recipe()
    candidate["steps"]["install"]["timeout_seconds"] = value
    with pytest.raises(ValueError, match="positive timeout"):
        validate_qualification_recipe(candidate)
