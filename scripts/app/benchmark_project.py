"""Local benchmark projects bind one decision workflow to configuration and evidence."""

import json
from pathlib import Path

from scripts.results.acceptance_policy import validate_policy
from scripts.app.benchmark_presets import build_portable_preset, validate_portable_preset
from scripts.results.result_store import atomic_write_json, validate_json_data


PROJECT_SCHEMA_VERSION = 1
PROJECT_WORKFLOWS = {
    "hardware_comparison": "Hardware comparison",
    "model_selection": "Model selection",
    "acceptance_validation": "Acceptance validation",
    "capacity_planning": "Capacity planning",
    "regression": "Regression",
}


def build_project(name: str, workflow: str, state: dict, *, baseline_result: str | None = None,
                  acceptance_policy: dict | None = None) -> dict:
    project = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "name": name.strip(),
        "workflow": workflow,
        "preset": build_portable_preset(name, state),
        "baseline_result": baseline_result,
        "acceptance_policy": acceptance_policy,
    }
    return validate_project(project)


def validate_project(project: dict) -> dict:
    fields = {"schema_version", "name", "workflow", "preset", "baseline_result", "acceptance_policy"}
    if not isinstance(project, dict) or set(project) != fields:
        raise ValueError("benchmark project is incomplete or contains unknown fields")
    if project["schema_version"] != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark-project schema: {project['schema_version']}")
    if not isinstance(project["name"], str) or not project["name"].strip():
        raise ValueError("benchmark project requires a name")
    if project["workflow"] not in PROJECT_WORKFLOWS:
        raise ValueError(f"unsupported benchmark-project workflow: {project['workflow']}")
    preset_errors = validate_portable_preset(project["preset"])
    if preset_errors:
        raise ValueError(preset_errors[0])
    baseline = project["baseline_result"]
    if baseline is not None and (not isinstance(baseline, str) or not baseline.strip()):
        raise ValueError("benchmark-project baseline must be a result path or null")
    if project["acceptance_policy"] is not None:
        validate_policy(project["acceptance_policy"])
    return project


def save_project(path: Path, project: dict) -> None:
    validate_project(project)
    atomic_write_json(Path(path), project)


def load_project(path: Path) -> dict:
    project = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_json_data(project)
    return validate_project(project)


def project_frontend_state(project: dict, local_options: dict) -> dict:
    validate_project(project)
    configuration = project["preset"]["configuration"]
    options = dict(local_options)
    options.update(configuration["options"])
    # No "engine": a project's preset never overrides the engines selected on screen.
    return {
        "tests": list(configuration["tests"]),
        "models": {key: list(values) for key, values in configuration["models"].items()},
        "max_prompt_tokens": configuration["max_prompt_tokens"],
        "tg_tokens": configuration["tg_tokens"], "gui_options": options,
    }
