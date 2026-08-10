from types import SimpleNamespace

import pytest

from scripts.setup.custom_models import load_custom_models
from scripts.setup.model_download import enough_disk_space, import_model
from scripts.setup.model_import import ImportVariant, inspect_repository


class FakeApi:
    def __init__(self, files):
        self.files = files

    def model_info(self, repo, **_kwargs):
        siblings = [SimpleNamespace(rfilename=name, size=size, lfs=None)
                    for name, size in self.files.items()]
        return SimpleNamespace(siblings=siblings, sha="commit", gated=False)


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
