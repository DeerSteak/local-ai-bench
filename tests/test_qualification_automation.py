import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.release.qualification_automation import (
    execution_recipe_gaps, finalize_qualification_run, initial_run_state,
    load_qualification_recipe, next_qualification_step,
    log_contains_marker, make_evidence_accessible, qualification_entry_from_run,
    qualification_preview, recipe_digest, sudo_invoking_owner, validate_qualification_recipe,
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
            "notes": "Smallest-model workload coverage; not performance qualification.",
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


def test_finalization_refuses_to_leave_a_pass_when_evidence_is_incomplete(monkeypatch, tmp_path):
    state = initial_run_state(recipe())
    for step in QUALIFICATION_LIFECYCLE:
        state["steps"][step]["status"] = "passed"
    monkeypatch.setattr(
        "scripts.release.qualification_automation.build_final_manifest",
        lambda *_args: (_ for _ in ()).throw(ValueError("target bundle missing")),
    )

    assert finalize_qualification_run(recipe(), state, tmp_path) is None
    assert state["steps"]["uninstall"]["status"] == "failed"
    assert state["steps"]["uninstall"]["detail"] == "target bundle missing"
    assert not (tmp_path / "qualification-manifest.json").exists()


def test_finalization_writes_and_verifies_the_pass_manifest(monkeypatch, tmp_path):
    state = initial_run_state(recipe())
    for step in QUALIFICATION_LIFECYCLE:
        state["steps"][step]["status"] = "passed"
    monkeypatch.setattr(
        "scripts.release.qualification_automation.build_final_manifest",
        lambda *_args: {"schema": "qualification-evidence-v1", "status": "passed"},
    )
    seen = []
    monkeypatch.setattr(
        "scripts.release.qualification_automation.verify_final_manifest",
        lambda path: seen.append(path),
    )

    manifest = finalize_qualification_run(recipe(), state, tmp_path)

    assert manifest == tmp_path / "qualification-manifest.json"
    assert seen == [tmp_path]


def test_evidence_permissions_are_readable_and_directories_traversable(tmp_path):
    output = tmp_path / "evidence"
    nested = output / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "result.json"
    artifact.write_text("{}")
    os.chmod(output, 0o700)
    os.chmod(nested, 0o700)
    os.chmod(artifact, 0o600)

    make_evidence_accessible(output, environ={}, effective_uid=501)

    assert output.stat().st_mode & 0o777 == 0o755
    assert nested.stat().st_mode & 0o777 == 0o755
    assert artifact.stat().st_mode & 0o777 == 0o644


def test_windows_evidence_access_never_resolves_posix_ownership_operations(tmp_path):
    calls = []
    make_evidence_accessible(
        tmp_path / "not-created", platform_name="nt",
        chmod=lambda *_args: calls.append("chmod"),
        chown=lambda *_args: calls.append("chown"),
    )
    assert calls == []
    assert not (tmp_path / "not-created").exists()


def test_qualification_automation_imports_when_os_chown_is_unavailable():
    command = (
        "import os; del os.chown; "
        "import scripts.release.qualification_automation"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_root_qualification_returns_evidence_to_sudo_invoking_user(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    artifact = output / "result.json"
    artifact.write_text("{}")
    ownership = []
    make_evidence_accessible(
        output, environ={"SUDO_UID": "1001", "SUDO_GID": "1002"}, effective_uid=0,
        chown=lambda path, uid, gid: ownership.append((path, uid, gid)),
    )
    assert ownership == [(output, 1001, 1002), (artifact, 1001, 1002)]


@pytest.mark.parametrize("environ", [{}, {"SUDO_UID": "bad", "SUDO_GID": "1002"}])
def test_invalid_or_absent_sudo_identity_is_never_used(environ):
    assert sudo_invoking_owner(environ, 0) is None


@pytest.mark.parametrize("value", [0, -1, True])
def test_timeouts_must_be_positive_integer_seconds(value):
    candidate = recipe()
    candidate["steps"]["install"]["timeout_seconds"] = value
    with pytest.raises(ValueError, match="positive timeout"):
        validate_qualification_recipe(candidate)
