from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.setup.custom_models import load_custom_models
from scripts.setup import model_download
from scripts.setup.model_download import (
    catalog_model_downloaded, catalog_mtp_artifact_download_size,
    catalog_mtp_artifact_downloaded,
    download_hf_files, download_hf_snapshot,
    enough_disk_space, import_model, provision_catalog_models,
)
from scripts.setup.model_import import ImportVariant, inspect_repository


class FakeApi:
    def __init__(self, files):
        self.files = files

    def model_info(self, repo, **_kwargs):
        siblings = [SimpleNamespace(rfilename=name, size=size, lfs=None)
                    for name, size in self.files.items()]
        return SimpleNamespace(siblings=siblings, sha="commit", gated=False)


def test_download_hf_files_uses_cli_and_flattens_nested_file(monkeypatch, tmp_path):
    destination = tmp_path / "models"
    source = destination / "nested" / "model.gguf"
    def run(_command, **_kwargs):
        source.parent.mkdir(parents=True)
        source.write_bytes(b"weights")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(model_download.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(model_download.subprocess, "run", run)

    assert download_hf_files("owner/model", "nested/model.gguf", destination)
    assert (destination / "model.gguf").read_bytes() == b"weights"
    assert not source.exists()


def test_download_hf_files_falls_back_to_python_api(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(model_download.shutil, "which", lambda _name: None)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: calls.append(kwargs))

    assert download_hf_files(
        "owner/model", ["one.gguf", "two.gguf"], tmp_path, token="secret",
    )
    assert [call["filename"] for call in calls] == ["one.gguf", "two.gguf"]
    assert all(call["token"] == "secret" for call in calls)


def test_download_hf_snapshot_reports_both_failures(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(model_download.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        model_download.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="cli failed", stdout=""),
    )
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("api failed")),
    )

    assert not download_hf_snapshot("owner/model", tmp_path, warn=warnings.append)
    assert warnings == ["hf error: cli failed", "Python API download failed: api failed"]


def test_catalog_model_downloaded_checks_llamacpp_files(monkeypatch, tmp_path):
    model = {"tag": "model", "hf_file": "model.gguf"}
    monkeypatch.setattr(model_download, "engine_model_complete", lambda *_args: True)

    assert catalog_model_downloaded(
        model, "llamacpp", models_dir=tmp_path / "models", vllm_cache=tmp_path / "cache",
    )


def test_catalog_mtp_artifact_downloaded_checks_only_separate_draft(tmp_path):
    embedded = {"tag": "embedded", "native_mtp": {
        "llamacpp": {"num_speculative_tokens": 3},
    }}
    separate = {"tag": "separate", "native_mtp": {"llamacpp": {
        "num_speculative_tokens": 3,
        "draft_repo": "owner/model", "draft_file": "MTP/draft.gguf",
    }}}
    assert catalog_mtp_artifact_downloaded(
        embedded, "llamacpp", models_dir=tmp_path,
    ) is True
    assert catalog_mtp_artifact_downloaded(
        separate, "llamacpp", models_dir=tmp_path,
    ) is False
    destination = tmp_path / "llamacpp" / "separate"
    destination.mkdir(parents=True)
    (destination / "draft.gguf").touch()
    assert catalog_mtp_artifact_downloaded(
        separate, "llamacpp", models_dir=tmp_path,
    ) is True


def test_catalog_mtp_artifact_download_size_is_only_for_separate_drafts():
    embedded = {"native_mtp": {"llamacpp": {"num_speculative_tokens": 3}}}
    separate = {"native_mtp": {"llamacpp": {
        "num_speculative_tokens": 3,
        "draft_repo": "owner/model", "draft_file": "MTP/draft.gguf",
        "draft_download_size": "~1.4 GB",
    }}}
    assert catalog_mtp_artifact_download_size(embedded, "llamacpp") is None
    assert catalog_mtp_artifact_download_size(separate, "llamacpp") == "~1.4 GB"


def test_provision_catalog_models_downloads_missing_llamacpp_model(monkeypatch, tmp_path):
    model = {
        "tag": "model", "label": "Model", "hf_repo": "owner/model",
        "hf_file": "model.gguf", "size": "1 GB",
    }
    downloads = []
    monkeypatch.setattr(model_download, "models_missing_engine_support", lambda *_args: [])
    monkeypatch.setattr(model_download, "engine_download_size", lambda *_args: "1 GB")
    monkeypatch.setattr(model_download, "catalog_model_downloaded", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        model_download, "download_hf_files",
        lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
    )

    provision_catalog_models(
        [model], ["llamacpp"], models_dir=tmp_path / "models",
        vllm_cache=tmp_path / "cache", load_token=lambda: "token", issues=[],
        info=lambda _msg: None, warn=lambda _msg: None,
        fail=lambda _msg: None, ok=lambda _msg: None,
    )

    assert downloads[0][0][0:2] == ("owner/model", "model.gguf")


def test_provision_downloads_shared_repo_variants_without_duplicate_vllm_snapshot(
        monkeypatch, tmp_path):
    variants = [
        {
            "tag": "demo:q4", "label": "Demo Q4", "variant": "Q4", "default": True,
            "hf_repo": "owner/demo", "hf_file": "demo-q4.gguf", "download_size": "~4 GB",
            "vllm_repo": "owner/demo-awq", "vllm_download_size": "~4 GB",
        },
        {
            "tag": "demo:q8", "label": "Demo Q8", "variant": "Q8",
            "hf_repo": "owner/demo", "hf_file": "demo-q8.gguf", "download_size": "~8 GB",
        },
    ]
    gguf_downloads, snapshots = [], []
    monkeypatch.setattr(model_download, "models_missing_engine_support", lambda *_args: [])
    monkeypatch.setattr(model_download, "catalog_model_downloaded", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        model_download, "download_hf_files",
        lambda repo, filename, *_args, **_kwargs: gguf_downloads.append((repo, filename)) or True,
    )
    monkeypatch.setattr(
        model_download, "download_hf_snapshot",
        lambda repo, *_args, **_kwargs: snapshots.append(repo) or True,
    )

    provision_catalog_models(
        variants, ["llamacpp", "vllm"], models_dir=tmp_path / "models",
        vllm_cache=tmp_path / "cache", load_token=lambda: None, issues=[],
        info=lambda _msg: None, warn=lambda _msg: None,
        fail=lambda _msg: None, ok=lambda _msg: None,
    )

    assert gguf_downloads == [
        ("owner/demo", "demo-q4.gguf"), ("owner/demo", "demo-q8.gguf"),
    ]
    assert snapshots == ["owner/demo-awq"]


def test_provision_catalog_models_downloads_only_missing_llamacpp_mtp_draft(
        monkeypatch, tmp_path):
    model = {
        "tag": "model", "label": "Model", "hf_repo": "owner/model",
        "hf_file": "model.gguf", "download_size": "~10 GB",
        "native_mtp": {"llamacpp": {
            "num_speculative_tokens": 3,
            "draft_repo": "owner/model", "draft_file": "MTP/draft.gguf",
            "draft_download_size": "~1 GB",
        }},
    }
    downloads = []
    monkeypatch.setattr(model_download, "models_missing_engine_support", lambda *_args: [])
    monkeypatch.setattr(model_download, "catalog_model_downloaded", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(model_download, "catalog_mtp_artifact_downloaded", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        model_download, "download_hf_files",
        lambda *args, **kwargs: downloads.append((args, kwargs)) or True,
    )

    provision_catalog_models(
        [model], ["llamacpp"], models_dir=tmp_path / "models",
        vllm_cache=tmp_path / "cache", load_token=lambda: "token", issues=[],
        info=lambda _msg: None, warn=lambda _msg: None,
        fail=lambda _msg: None, ok=lambda _msg: None,
    )

    assert downloads[0][0][:2] == ("owner/model", "MTP/draft.gguf")


def test_llamacpp_import_downloads_selected_files_and_registers(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model-Q4.gguf": 10}))
    downloaded = []

    def download(**kwargs):
        destination = tmp_path / "models" / "llamacpp" / "custom" / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"gguf")
        downloaded.append(kwargs)
        return str(destination)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    registry = tmp_path / "registry.json"
    record = import_model(
        inspection=inspection, engine="llamacpp", variant=inspection.llama_variants[0],
        tag="custom", label="Custom", models_dir=tmp_path / "models", registry_path=registry,
    )

    assert downloaded[0]["revision"] == "commit"
    assert record["format"] == "gguf"
    assert load_custom_models(registry) == [record]


def test_vllm_import_uses_cache_and_rejects_duplicate_tag(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors": 10,
    }))
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        snapshot = tmp_path / "cache" / "hub" / "models--owner--model" / "snapshots" / "commit"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"weights")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    registry = tmp_path / "registry.json"
    variant = inspection.vllm_variant
    assert variant is not None
    import_model(
        inspection=inspection, engine="vllm", variant=variant, tag="custom",
        label="Custom", vllm_cache=tmp_path / "cache", registry_path=registry,
    )
    assert calls[0]["cache_dir"] == str(tmp_path / "cache" / "hub")
    assert "model.safetensors" in calls[0]["allow_patterns"]
    assert calls[0]["allow_patterns"] == ["model.safetensors", "config.json"]
    with pytest.raises(ValueError, match="already registered"):
        import_model(
            inspection=inspection, engine="vllm", variant=variant, tag="custom",
            label="Again", vllm_cache=tmp_path / "cache", registry_path=registry,
        )


def test_disk_check_handles_known_size(tmp_path):
    tiny = ImportVariant("q4", "Q4", ("model.gguf",), 1)
    assert enough_disk_space(tiny, tmp_path / "new") is True


def test_failed_llamacpp_import_preserves_preexisting_empty_destination(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 10}))
    destination = tmp_path / "models" / "llamacpp" / "custom"
    destination.mkdir(parents=True)
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **_kwargs: (_ for _ in ()).throw(OSError("fail")),
    )

    with pytest.raises(OSError, match="fail"):
        import_model(
            inspection=inspection, engine="llamacpp", variant=inspection.llama_variants[0],
            tag="custom", label="Custom", models_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
        )

    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_cancelled_llamacpp_import_cleans_destination_and_does_not_register(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 10}))
    cancelled = [False]

    def download(**kwargs):
        destination = tmp_path / "models" / "llamacpp" / "custom" / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"partial")
        cancelled[0] = True
        return str(destination)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    registry = tmp_path / "registry.json"
    with pytest.raises(InterruptedError, match="cancelled"):
        import_model(
            inspection=inspection, engine="llamacpp", variant=inspection.llama_variants[0],
            tag="custom", label="Custom", models_dir=tmp_path / "models",
            registry_path=registry, cancel_check=lambda: cancelled[0],
        )
    assert not (tmp_path / "models" / "llamacpp" / "custom").exists()
    assert load_custom_models(registry) == []


def test_last_instant_llamacpp_cancel_cleans_destination_and_allows_retry(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 10}))
    calls = [0]

    def cancel_after_download_checks():
        calls[0] += 1
        return calls[0] == 3

    def download(**kwargs):
        destination = tmp_path / "models" / "llamacpp" / "custom" / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"complete")
        return str(destination)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    registry = tmp_path / "registry.json"
    arguments = {
        "inspection": inspection, "engine": "llamacpp",
        "variant": inspection.llama_variants[0], "tag": "custom", "label": "Custom",
        "models_dir": tmp_path / "models", "registry_path": registry,
    }
    with pytest.raises(InterruptedError, match="cancelled"):
        import_model(**arguments, cancel_check=cancel_after_download_checks)

    assert not (tmp_path / "models" / "llamacpp" / "custom").exists()
    assert load_custom_models(registry) == []
    record = import_model(**arguments)
    assert load_custom_models(registry) == [record]


def test_cancelled_vllm_import_removes_partial_cache_and_does_not_register(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors": 10,
    }))
    cancelled = [False]

    repo_cache = tmp_path / "cache" / "hub" / "models--owner--model"
    existing_blob = repo_cache / "blobs" / "complete"
    existing_blob.parent.mkdir(parents=True)
    existing_blob.write_bytes(b"complete")

    def snapshot_download(**_kwargs):
        incomplete = repo_cache / "blobs" / "weights.incomplete"
        incomplete.write_bytes(b"partial")
        snapshot = repo_cache / "snapshots" / "commit"
        snapshot.mkdir(parents=True)
        (snapshot / "model.safetensors").symlink_to(incomplete)
        lock = tmp_path / "cache" / "hub" / ".locks" / "models--owner--model" / "weights.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text("", encoding="utf-8")
        cancelled[0] = True

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    registry = tmp_path / "registry.json"
    assert inspection.vllm_variant is not None
    with pytest.raises(InterruptedError, match="cancelled"):
        import_model(
            inspection=inspection, engine="vllm", variant=inspection.vllm_variant,
            tag="custom", label="Custom", vllm_cache=tmp_path / "cache",
            registry_path=registry, cancel_check=lambda: cancelled[0],
        )
    assert load_custom_models(registry) == []
    assert existing_blob.read_bytes() == b"complete"
    assert sorted(path.relative_to(repo_cache) for path in repo_cache.rglob("*")) == [
        Path("blobs"), Path("blobs/complete"),
    ]
    assert not (tmp_path / "cache" / "hub" / ".locks" / "models--owner--model").exists()


def test_cancelled_vllm_import_removes_new_repo_cache(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors": 10,
    }))
    cancelled = [False]
    repo_cache = tmp_path / "cache" / "hub" / "models--owner--model"

    def snapshot_download(**_kwargs):
        partial = repo_cache / "blobs" / "weights.incomplete"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"partial")
        cancelled[0] = True

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert inspection.vllm_variant is not None
    with pytest.raises(InterruptedError, match="cancelled"):
        import_model(
            inspection=inspection, engine="vllm", variant=inspection.vllm_variant,
            tag="custom", label="Custom", vllm_cache=tmp_path / "cache",
            registry_path=tmp_path / "registry.json", cancel_check=lambda: cancelled[0],
        )
    assert not repo_cache.exists()


def test_failed_llamacpp_import_clears_partial_download_cache_for_retry(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 10}))
    destination = tmp_path / "models" / "llamacpp" / "custom"
    destination.mkdir(parents=True)

    def partial_download(**_kwargs):
        partial = destination / ".cache" / "huggingface" / "download" / "partial"
        partial.mkdir(parents=True)
        (partial / "model.gguf.part").write_bytes(b"partial")
        raise OSError("network lost")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", partial_download)
    with pytest.raises(OSError, match="network lost"):
        import_model(
            inspection=inspection, engine="llamacpp", variant=inspection.llama_variants[0],
            tag="custom", label="Custom", models_dir=tmp_path / "models",
            registry_path=tmp_path / "registry.json",
        )

    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_import_replaces_registration_after_artifacts_were_deleted(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/new", api=FakeApi({"model.gguf": 10}))
    registry = tmp_path / "registry.json"
    from scripts.setup.custom_models import save_custom_model
    save_custom_model({
        "engine": "llamacpp", "tag": "custom", "repo": "owner/old",
        "files": ["old.gguf"],
    }, registry)

    def download(**kwargs):
        target = tmp_path / "models" / "llamacpp" / "custom" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"gguf")
        return str(target)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    record = import_model(
        inspection=inspection, engine="llamacpp", variant=inspection.llama_variants[0],
        tag="custom", label="New", models_dir=tmp_path / "models", registry_path=registry,
    )

    assert record["repo"] == "owner/new"
    assert load_custom_models(registry) == [record]


def test_failed_stale_vllm_reimport_preserves_registration(tmp_path):
    inspection = inspect_repository("owner/new", api=FakeApi({
        "config.json": 1, "model.safetensors": 10,
    }))
    registry = tmp_path / "registry.json"
    previous = {
        "engine": "vllm", "tag": "custom", "repo": "owner/old",
        "format": "safetensors", "files": [],
    }
    from scripts.setup.custom_models import save_custom_model
    save_custom_model(previous, registry)
    variant = inspection.vllm_variant
    assert variant is not None

    with pytest.raises(ValueError, match="cache location is unavailable"):
        import_model(
            inspection=inspection, engine="vllm", variant=variant, tag="custom",
            label="New", vllm_cache=None, registry_path=registry,
        )

    assert load_custom_models(registry) == [previous]
