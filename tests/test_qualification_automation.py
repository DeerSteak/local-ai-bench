import json

import pytest

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.release.qualification_automation import (
    load_qualification_recipe, qualification_preview, validate_qualification_recipe,
)


def recipe():
    steps = {
        name: {
            "command": ["tool", name], "timeout_seconds": 60,
            "expected_exit_codes": [0], "interrupt_after_seconds": None,
        }
        for name in QUALIFICATION_LIFECYCLE
    }
    steps["cancellation"].update({
        "expected_exit_codes": [130], "interrupt_after_seconds": 5,
    })
    return {
        "target": {
            "id": "macos-arm64-llamacpp-metal", "platform": "macos",
            "architecture": "arm64", "runtime": "llamacpp",
            "runtime_version": "b6000", "backend": "metal",
        },
        "coverage": {
            "workloads": ["llm"], "models": ["llama3.2:3b-instruct-q4_K_M"],
            "notes": "Lifecycle smoke; not a full catalog performance qualification.",
        },
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
    value["steps"]["resume"]["interrupt_after_seconds"] = 2
    with pytest.raises(ValueError, match="only the cancellation"):
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


@pytest.mark.parametrize("value", [0, -1, True])
def test_timeouts_must_be_positive_integer_seconds(value):
    candidate = recipe()
    candidate["steps"]["install"]["timeout_seconds"] = value
    with pytest.raises(ValueError, match="positive timeout"):
        validate_qualification_recipe(candidate)
