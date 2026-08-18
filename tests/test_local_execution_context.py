import json
import os
from pathlib import Path

import pytest

from scripts.results.local_execution_context import (
    LocalExecutionContext, load_local_execution_context, local_execution_path,
    write_local_execution_context,
)


def test_private_context_round_trips_absolute_paths_with_owner_only_mode(tmp_path):
    event_path = tmp_path / "result.events.sqlite3"
    context = LocalExecutionContext(
        "job_example", (tmp_path / "Custom ComfyUI").resolve(),
        (tmp_path / "generated images").resolve(),
    )
    path = write_local_execution_context(event_path, context)
    assert path == local_execution_path(event_path)
    assert load_local_execution_context(event_path, "job_example") == context
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_private_context_rejects_relative_paths_unknown_fields_and_wrong_job(tmp_path):
    with pytest.raises(ValueError, match="absolute comfyui_dir"):
        LocalExecutionContext("job_example", Path("relative"), tmp_path.resolve())
    event_path = tmp_path / "result.events.sqlite3"
    path = local_execution_path(event_path)
    path.write_text(json.dumps({
        "schema_version": 1, "job_id": "job_example",
        "comfyui_dir": str(tmp_path.resolve()), "images_dir": str(tmp_path.resolve()),
        "extra": "not allowed",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_local_execution_context(event_path, "job_example")
    path.write_text(json.dumps({
        "schema_version": 1, "job_id": "job_other",
        "comfyui_dir": str(tmp_path.resolve()), "images_dir": str(tmp_path.resolve()),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="different job"):
        load_local_execution_context(event_path, "job_example")


def test_private_context_missing_or_malformed_is_a_stable_validation_error(tmp_path):
    event_path = tmp_path / "result.events.sqlite3"
    with pytest.raises(ValueError, match="missing"):
        load_local_execution_context(event_path, "job_example")
    local_execution_path(event_path).write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_local_execution_context(event_path, "job_example")
