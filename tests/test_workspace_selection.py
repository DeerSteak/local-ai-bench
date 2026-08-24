import json

import pytest

from scripts.results.workspace_selection import (
    build_workspace_selection, load_workspace_selection, validate_workspace_selection,
)


def result_file(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps({"value": value}), encoding="utf-8")
    return path


def test_selection_records_distinct_content_identities_and_baseline(tmp_path):
    first = result_file(tmp_path, "first.json", 1)
    second = result_file(tmp_path, "second.json", 2)
    selection = build_workspace_selection(
        [first, second], baseline_path=second,
        view={"section": "llm", "enabled_models": ["model"]},
    )
    assert [item["name"] for item in selection["results"]] == ["first.json", "second.json"]
    assert selection["baseline_sha256"] == selection["results"][1]["sha256"]
    assert all(len(item["sha256"]) == 64 for item in selection["results"])
    assert not any(str(tmp_path) in json.dumps(item) for item in selection["results"])


def test_selection_round_trips_embedded_policy(tmp_path):
    result = result_file(tmp_path, "result.json", 1)
    policy = {
        "schema_version": 1, "name": "Gate", "methodology_profile": "neutral-v1",
        "rules": [{
            "id": "tps", "section": "llm", "model": "model", "case": "2K",
            "metric": "tps_mean", "operator": "at_least", "threshold": 10,
            "minimum_evidence": 1,
        }],
    }
    selection = build_workspace_selection([result], policy=policy)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    assert load_workspace_selection(path) == selection


def test_selection_rejects_duplicate_content_and_foreign_baseline(tmp_path):
    first = result_file(tmp_path, "first.json", 1)
    duplicate = result_file(tmp_path, "duplicate.json", 1)
    other = result_file(tmp_path, "other.json", 2)
    with pytest.raises(ValueError, match="distinct"):
        build_workspace_selection([first, duplicate])
    with pytest.raises(ValueError, match="baseline"):
        build_workspace_selection([first], baseline_path=other)


def test_selection_rejects_paths_unknown_view_and_bad_digest(tmp_path):
    result = result_file(tmp_path, "result.json", 1)
    with pytest.raises(ValueError, match="unknown fields"):
        build_workspace_selection([result], view={"private_path": str(tmp_path)})
    selection = build_workspace_selection([result])
    selection["results"][0]["sha256"] = "short"
    with pytest.raises(ValueError, match="identity"):
        validate_workspace_selection(selection)
