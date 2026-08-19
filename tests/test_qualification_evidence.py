import json
from pathlib import Path

import pytest

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.release.qualification_coverage import qualification_workloads
from scripts.release.qualification_evidence import (
    archive_generated_artifacts, file_identity, final_evidence_errors,
    installation_inventory, source_inventory,
    verify_final_manifest,
)


def repo(tmp_path):
    (tmp_path / "README.md").touch()
    (tmp_path / "scripts").mkdir()
    return tmp_path


def test_file_identity_records_exact_content(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"qualification")
    assert file_identity(artifact) == {
        "sha256": "1dbf39600b5761d58378447f494a50c8b9c01b559b6ef420720f99f4e45717c9",
        "size": 13,
    }


def test_installation_inventory_hashes_runtime_models_and_sources(tmp_path):
    root = repo(tmp_path)
    runtime = root / "llama.cpp"
    runtime.mkdir()
    (runtime / "llama-server").write_bytes(b"runtime")
    models = root / "models"
    models.mkdir()
    (models / "google_gemma-3-1b-it-Q4_K_M.gguf").write_bytes(b"model")

    inventory = installation_inventory(
        root, "llamacpp", "b10488", "gemma3:1b-it-q4_K_M", include_models=True,
    )

    assert inventory["version"] == "b10488"
    assert inventory["runtime_files"][0]["path"] == "llama.cpp/llama-server"
    assert inventory["model_files"][0]["path"].startswith("models/")
    assert {item["id"] for item in inventory["model_sources"]} == {
        "gemma3:1b-it-q4_K_M", "nomic-embed-text", "sd15",
    }


def test_installation_inventory_refuses_missing_identity_artifacts(tmp_path):
    root = repo(tmp_path)
    (root / "llama.cpp").mkdir()
    with pytest.raises(ValueError, match="identity-bearing executable"):
        installation_inventory(
            root, "llamacpp", "b1", "gemma3:1b-it-q4_K_M", include_models=False,
        )


def test_generated_images_are_copied_into_evidence(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("{}")
    images = tmp_path / "external-images"
    images.mkdir()
    (images / "sd15_512x512.png").write_bytes(b"png")
    result.with_suffix(".events.sqlite3.local.json").write_text(json.dumps({
        "images_dir": str(images),
    }))

    copied = archive_generated_artifacts(result, tmp_path / "evidence-images")

    assert [path.name for path in copied] == ["sd15_512x512.png"]
    assert copied[0].read_bytes() == b"png"


def test_source_inventory_records_commit_and_ignores_untracked_files(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    (tmp_path / "tracked").write_text("ok")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "test"], check=True)
    (tmp_path / "untracked").write_text("ignored")

    inventory = source_inventory(tmp_path)

    assert inventory["commit"]["status"] == "captured"
    assert inventory["tracked_worktree_dirty"] is False


def test_final_gate_reports_every_missing_evidence_class(tmp_path, monkeypatch):
    state = {"steps": {
        name: {"status": "passed", "log": f"{name}.log"} for name in QUALIFICATION_LIFECYCLE
    }}
    recipe = {
        "target": {
            "runtime": "vllm", "runtime_version": "1.1", "platform": "linux",
            "architecture": "x86_64", "backend": "cuda", "accelerator": "GPU",
        },
        "steps": {"install": {"command": [
            "tool", "--root", str(tmp_path), "--version", "1.0",
        ]}},
    }
    monkeypatch.setattr(
        "scripts.release.qualification_evidence.source_inventory",
        lambda _root: {"commit": {"status": "captured"}, "tracked_worktree_dirty": False},
    )
    monkeypatch.setattr(
        "scripts.release.qualification_evidence.host_inventory",
        lambda: {"probes": {
            "kernel": {"status": "captured"}, "accelerator_driver": {"status": "captured"},
        }},
    )

    errors = final_evidence_errors(recipe, state, tmp_path)

    assert "baseline workload result is missing" in errors
    assert "target workload result is missing" in errors
    assert "cancelled run was not successfully resumed to completion" in errors
    assert "baseline report is missing" in errors
    assert "target bundle is missing" in errors
    assert "baseline installation inventory is missing" in errors
    assert all(f"lifecycle log is missing: {name}" in errors for name in QUALIFICATION_LIFECYCLE)


def test_final_manifest_verifier_rejects_missing_and_modified_files(tmp_path):
    artifact = tmp_path / "result.json"
    artifact.write_text("original")
    manifest = {
        "schema": "qualification-evidence-v1", "status": "passed",
        "files": {"result.json": file_identity(artifact)},
    }
    (tmp_path / "qualification-manifest.json").write_text(json.dumps(manifest))
    assert verify_final_manifest(tmp_path) == manifest
    artifact.write_text("modified")
    with pytest.raises(ValueError, match="integrity check failed"):
        verify_final_manifest(tmp_path)
