"""Resumable execution plans for platform qualification."""

import json
from pathlib import Path

from scripts.release.qualification import QUALIFICATION_LIFECYCLE


RECIPE_KEYS = {"target", "coverage", "steps"}
TARGET_KEYS = {"id", "platform", "architecture", "runtime", "runtime_version", "backend"}
COVERAGE_KEYS = {"workloads", "models", "notes"}
STEP_KEYS = {"command", "timeout_seconds", "expected_exit_codes", "interrupt_after_seconds"}


def validate_qualification_recipe(recipe: dict) -> None:
    if not isinstance(recipe, dict) or set(recipe) != RECIPE_KEYS:
        raise ValueError("qualification recipe has missing or unknown fields")
    target = recipe["target"]
    if not isinstance(target, dict) or set(target) != TARGET_KEYS:
        raise ValueError("qualification recipe target has missing or unknown fields")
    if any(not isinstance(target[key], str) or not target[key].strip() for key in TARGET_KEYS):
        raise ValueError("qualification recipe target fields must be non-empty text")
    coverage = recipe["coverage"]
    if not isinstance(coverage, dict) or set(coverage) != COVERAGE_KEYS:
        raise ValueError("qualification recipe coverage has missing or unknown fields")
    for key in ("workloads", "models"):
        if (not isinstance(coverage[key], list) or not coverage[key]
                or any(not isinstance(item, str) or not item.strip() for item in coverage[key])):
            raise ValueError(f"qualification recipe coverage requires non-empty {key}")
    if not isinstance(coverage["notes"], str):
        raise ValueError("qualification recipe coverage notes must be text")
    steps = recipe["steps"]
    if not isinstance(steps, dict) or set(steps) != set(QUALIFICATION_LIFECYCLE):
        raise ValueError("qualification recipe must define every lifecycle step")
    for name, step in steps.items():
        if not isinstance(step, dict) or set(step) != STEP_KEYS:
            raise ValueError(f"qualification step {name} has missing or unknown fields")
        command = step["command"]
        if (not isinstance(command, list) or not command
                or any(not isinstance(arg, str) or not arg for arg in command)):
            raise ValueError(f"qualification step {name} requires an argv command")
        timeout = step["timeout_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise ValueError(f"qualification step {name} requires a positive timeout")
        exit_codes = step["expected_exit_codes"]
        if (not isinstance(exit_codes, list) or not exit_codes
                or any(not isinstance(code, int) or isinstance(code, bool) for code in exit_codes)):
            raise ValueError(f"qualification step {name} requires expected exit codes")
        interrupt = step["interrupt_after_seconds"]
        if interrupt is not None and (
                not isinstance(interrupt, int) or isinstance(interrupt, bool) or interrupt < 1
                or interrupt >= timeout):
            raise ValueError(f"qualification step {name} has an invalid interrupt delay")
        if (name == "cancellation") != (interrupt is not None):
            raise ValueError("only the cancellation step may define an interrupt delay")


def load_qualification_recipe(path: Path) -> dict:
    try:
        recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read qualification recipe: {exc}") from None
    validate_qualification_recipe(recipe)
    return recipe


def qualification_preview(recipe: dict, output_dir: Path) -> dict:
    validate_qualification_recipe(recipe)
    output_dir = Path(output_dir).resolve()
    return {
        "mode": "preview",
        "target": dict(recipe["target"]),
        "coverage": dict(recipe["coverage"]),
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "qualification-state.json"),
        "steps": [
            {
                "name": name,
                "command": list(recipe["steps"][name]["command"]),
                "timeout_seconds": recipe["steps"][name]["timeout_seconds"],
                "interrupt_after_seconds": recipe["steps"][name]["interrupt_after_seconds"],
            }
            for name in QUALIFICATION_LIFECYCLE
        ],
    }
