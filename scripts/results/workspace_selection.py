"""Portable selection state shared by workspace views and exports."""

import hashlib
import json
from pathlib import Path

from scripts.results.acceptance_policy import validate_policy
from scripts.results.result_store import validate_json_data


WORKSPACE_SELECTION_SCHEMA_VERSION = 1
VIEW_FIELDS = {
    "section", "accuracy_test", "enabled_models", "enabled_image_models",
    "enabled_embedding_models", "hostname_overrides",
}


def result_identity(path: Path) -> dict:
    source = Path(path)
    return {
        "name": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def build_workspace_selection(result_paths: list[Path], *, baseline_path: Path | None = None,
                              policy: dict | None = None,
                              recommendation: dict | None = None,
                              view: dict | None = None) -> dict:
    if not result_paths:
        raise ValueError("workspace selection requires at least one result")
    identities = [result_identity(path) for path in result_paths]
    if len({item["sha256"] for item in identities}) != len(identities):
        raise ValueError("workspace selection results must be distinct")
    baseline = result_identity(baseline_path)["sha256"] if baseline_path else None
    if baseline is not None and baseline not in {item["sha256"] for item in identities}:
        raise ValueError("workspace baseline must be one of the selected results")
    if policy is not None:
        validate_policy(policy)
    if recommendation is not None and recommendation.get("artifact_type") != "recommendation":
        raise ValueError("workspace recommendation artifact is invalid")
    selected_view = dict(view or {})
    unknown = set(selected_view) - VIEW_FIELDS
    if unknown:
        raise ValueError(f"workspace view contains unknown fields: {', '.join(sorted(unknown))}")
    selection = {
        "artifact_type": "workspace_selection",
        "schema_version": WORKSPACE_SELECTION_SCHEMA_VERSION,
        "results": identities,
        "baseline_sha256": baseline,
        "view": selected_view,
        "acceptance_policy": policy,
        "recommendation": recommendation,
    }
    validate_workspace_selection(selection)
    return selection


def validate_workspace_selection(selection: dict) -> dict:
    if not isinstance(selection, dict) or selection.get("artifact_type") != "workspace_selection" \
            or selection.get("schema_version") != WORKSPACE_SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported workspace-selection artifact")
    if set(selection) != {
        "artifact_type", "schema_version", "results", "baseline_sha256", "view",
        "acceptance_policy", "recommendation",
    }:
        raise ValueError("workspace selection contains unknown fields")
    results = selection.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("workspace selection requires results")
    digests = []
    for item in results:
        if not isinstance(item, dict) or set(item) != {"name", "sha256"} \
                or not isinstance(item["name"], str) or not item["name"] \
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ValueError("workspace selection result identity is invalid")
        digests.append(item["sha256"])
    if len(set(digests)) != len(digests):
        raise ValueError("workspace selection results must be distinct")
    baseline = selection.get("baseline_sha256")
    if baseline is not None and baseline not in digests:
        raise ValueError("workspace baseline must be one of the selected results")
    view = selection.get("view")
    if not isinstance(view, dict) or set(view) - VIEW_FIELDS:
        raise ValueError("workspace view is invalid")
    if selection.get("acceptance_policy") is not None:
        validate_policy(selection["acceptance_policy"])
    recommendation = selection.get("recommendation")
    if recommendation is not None and recommendation.get("artifact_type") != "recommendation":
        raise ValueError("workspace recommendation artifact is invalid")
    validate_json_data(selection)
    return selection


def load_workspace_selection(path: Path) -> dict:
    return validate_workspace_selection(json.loads(Path(path).read_text(encoding="utf-8")))
