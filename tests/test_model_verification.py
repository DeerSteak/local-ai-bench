from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.results.model_verification import verify_resume_model
from scripts.results.resume_policy import file_identity
from scripts.runtime import config


def journal_for(artifacts, *, recovery=True):
    return SimpleNamespace(
        plan=SimpleNamespace(job_id="job"),
        store=SimpleNamespace(
            events=lambda _: [SimpleNamespace(payload={"recovery": "resume"})] if recovery else [],
            resume_identity=lambda _: {"artifacts": artifacts},
        ),
    )


def test_only_requested_model_is_hashed_once_per_stage(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"weights")
    saved = {"model:m:part1": file_identity(path), "model:other:part1": {"sha256": "untouched"}}
    journal = journal_for(saved)
    engine = SimpleNamespace(model_pulled=lambda _: True, resume_artifact_paths=lambda _: (path,))
    progress = Mock()
    verify_resume_model(journal, {"tag": "m", "short": "m"}, engine, progress=progress)
    assert progress.call_count == 2
    verify_resume_model(journal, {"tag": "m", "short": "m"}, engine, progress=progress)
    assert progress.call_count == 2


@pytest.mark.parametrize("change", ["bytes", "missing", "extra"])
def test_model_drift_is_rejected_before_load(tmp_path, change):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"original")
    journal = journal_for({"model:m:part1": file_identity(path)})
    paths = (path,)
    if change == "bytes":
        path.write_bytes(b"modified")
    elif change == "missing":
        paths = ()
    else:
        paths = (path, path)
    engine = SimpleNamespace(model_pulled=lambda _: bool(paths), resume_artifact_paths=lambda _: paths)
    with pytest.raises(ValueError, match="create a fork"):
        verify_resume_model(journal, {"tag": "m", "short": "m"}, engine)
    assert not getattr(journal, "_verified_model_weights", set())


def test_later_model_is_checked_when_reached(tmp_path):
    paths = {name: tmp_path / name for name in ("first", "later")}
    for path in paths.values():
        path.write_bytes(b"original")
    journal = journal_for({f"model:{name}:part1": file_identity(path) for name, path in paths.items()})
    engine = SimpleNamespace(model_pulled=lambda _: True, resume_artifact_paths=lambda tag: (paths[tag],))
    verify_resume_model(journal, {"tag": "first", "short": "first"}, engine)
    paths["later"].write_bytes(b"modified")
    with pytest.raises(ValueError, match="later"):
        verify_resume_model(journal, {"tag": "later", "short": "later"}, engine)


def test_fresh_run_and_non_journal_work_do_not_duplicate_initial_hashes():
    engine = Mock()
    for journal in (None, journal_for({}, recovery=False)):
        verify_resume_model(journal, {"tag": "m", "short": "m"}, engine)
    engine.resume_artifact_paths.assert_not_called()


def test_image_support_assets_are_verified(tmp_path, monkeypatch):
    from scripts.workloads.image_benchmark import image_resume_artifacts

    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", tmp_path)
    model = {"short": "image", "workflow": "sdxl", "checkpoint": "model.safetensors",
             "support_assets": [{"folder": "vae", "name": "encoder.safetensors"}]}
    for folder, name in (("checkpoints", "model.safetensors"), ("vae", "encoder.safetensors")):
        path = tmp_path / folder / name
        path.parent.mkdir()
        path.write_bytes(b"original")
    saved = {name: file_identity(path) for name, path in image_resume_artifacts([model]).items()}
    journal = journal_for(saved)
    (tmp_path / "vae" / "encoder.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="create a fork"):
        verify_resume_model(journal, model)
