import pytest

from model_inventory import (
    build_model_inventory,
    classify_engine_models,
    delete_non_catalog_model_dirs,
    find_non_catalog_model_dirs,
    format_model_inventory,
    installed_image_models,
    model_tag_slug,
    sanitize_tag_to_short,
)


LLM_CATALOG = [
    {"tag": "llm-small", "label": "Small LLM", "short": "small", "tier": "small"},
    {"tag": "llm-large", "label": "Large LLM", "short": "large", "tier": "large"},
]
EMBED_CATALOG = [
    {"tag": "embed-one", "label": "Embed One", "short": "embed-one"},
]
IMAGE_CATALOG = [
    {"short": "image-one", "label": "Image One", "checkpoint": "one.safetensors"},
    {"short": "image-two", "label": "Image Two", "checkpoint": "two.safetensors"},
]


class FakeEngine:
    name = "fake"

    def __init__(self, installed):
        self.installed = installed
        self.list_calls = 0

    def list_installed_models(self):
        self.list_calls += 1
        return self.installed


def test_classifies_catalog_and_custom_models_in_catalog_order():
    inventory = classify_engine_models(
        [
            {"tag": "custom-folder", "size": 9},
            {"tag": "embed-one", "size": 7},
            {"tag": "llm-large", "size": 5},
            {"tag": "llm-small", "size": 3},
        ],
        llm_catalog=LLM_CATALOG,
        embed_catalog=EMBED_CATALOG,
    )

    assert [model["tag"] for model in inventory["llm"]] == ["llm-small", "llm-large"]
    assert [model["tag"] for model in inventory["embedding"]] == ["embed-one"]
    assert inventory["custom"] == [{
        "tag": "custom-folder",
        "label": "custom-folder (custom)",
        "short": "custom-folder",
        "size": 9,
    }]


def test_classification_omits_uninstalled_catalog_models():
    inventory = classify_engine_models(
        [{"tag": "llm-small", "size": 3}],
        llm_catalog=LLM_CATALOG,
        embed_catalog=EMBED_CATALOG,
    )

    assert [model["tag"] for model in inventory["llm"]] == ["llm-small"]
    assert inventory["embedding"] == []
    assert inventory["custom"] == []


def test_custom_models_are_sorted_by_folder_name():
    inventory = classify_engine_models(
        [{"tag": "z-custom", "size": 2}, {"tag": "a-custom", "size": 1}],
        llm_catalog=[],
        embed_catalog=[],
    )

    assert [model["tag"] for model in inventory["custom"]] == ["a-custom", "z-custom"]


def test_installed_images_use_explicit_comfyui_path(tmp_path):
    checkpoints = tmp_path / "models" / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "two.safetensors"
    checkpoint.write_bytes(b"12345")

    installed = installed_image_models(tmp_path, IMAGE_CATALOG)

    assert [model["short"] for model in installed] == ["image-two"]
    assert installed[0]["path"] == checkpoint
    assert installed[0]["size"] == 5


def test_installed_images_empty_when_checkpoint_directory_missing(tmp_path):
    assert installed_image_models(tmp_path, IMAGE_CATALOG) == []


def test_build_inventory_reads_engine_once_and_adds_images(monkeypatch, tmp_path):
    engine = FakeEngine([{"tag": LLM_CATALOG[0]["tag"], "size": 3}])
    monkeypatch.setattr("model_inventory.LLM_MODELS", LLM_CATALOG)
    monkeypatch.setattr("model_inventory.EMBED_MODELS", EMBED_CATALOG)
    monkeypatch.setattr("model_inventory.IMAGE_MODELS", IMAGE_CATALOG)
    checkpoints = tmp_path / "models" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "one.safetensors").write_bytes(b"1")

    inventory = build_model_inventory(engine, tmp_path)

    assert engine.list_calls == 1
    assert [model["tag"] for model in inventory["llm"]] == ["llm-small"]
    assert [model["short"] for model in inventory["image"]] == ["image-one"]


def test_format_inventory_groups_every_family():
    inventory = {
        "llm": [{"tag": "llm-one", "label": "LLM One", "size": 1_000_000_000}],
        "embedding": [{"tag": "embed-one", "label": "Embed One", "size": 500_000_000}],
        "custom": [{"tag": "custom-one", "label": "custom-one (custom)", "size": None}],
        "image": [{"short": "image-one", "label": "Image One", "size": 2_000_000_000}],
    }

    text = "\n".join(format_model_inventory(inventory, "fake"))

    assert "Downloaded models (fake)" in text
    assert "LLM:" in text
    assert "Embeddings:" in text
    assert "Custom LLM:" in text
    assert "Image generation:" in text
    assert "1 LLM, 1 embedding, 1 custom, 1 image installed" in text


def test_format_inventory_handles_every_group_empty():
    lines = format_model_inventory(
        {"llm": [], "embedding": [], "custom": [], "image": []}, "fake",
    )
    assert lines[-1] == "  0 LLM, 0 embedding, 0 custom, 0 image installed"


def test_sanitize_tag_to_short_replaces_tag_separators():
    assert sanitize_tag_to_short("org/model:latest") == "org-model-latest"


def test_model_tag_slug_matches_llamacpp_directory_naming():
    assert model_tag_slug("org/model:latest") == "org_model_latest"


def test_find_non_catalog_model_dirs_is_sorted_and_ignores_files(tmp_path):
    (tmp_path / "llm-small").mkdir()
    (tmp_path / "embed-one").mkdir()
    (tmp_path / "z-custom").mkdir()
    (tmp_path / "a-custom").mkdir()
    (tmp_path / "z-custom" / "model.gguf").write_bytes(b"model")
    (tmp_path / "a-custom" / "model.gguf").write_bytes(b"model")
    (tmp_path / "unrelated-folder").mkdir()
    (tmp_path / "loose.gguf").write_bytes(b"model")

    found = find_non_catalog_model_dirs(
        tmp_path, llm_catalog=LLM_CATALOG, embed_catalog=EMBED_CATALOG,
    )

    assert [path.name for path in found] == ["a-custom", "z-custom"]


def test_find_non_catalog_model_dirs_handles_missing_root(tmp_path):
    assert find_non_catalog_model_dirs(
        tmp_path / "missing", llm_catalog=LLM_CATALOG, embed_catalog=EMBED_CATALOG,
    ) == []


def test_delete_non_catalog_model_dirs_removes_only_explicit_safe_names(tmp_path):
    catalog = tmp_path / "llm-small"
    custom = tmp_path / "custom-model"
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    catalog.mkdir()
    custom.mkdir()
    (custom / "model.gguf").write_bytes(b"model")
    outside.mkdir()

    removed, failures = delete_non_catalog_model_dirs(
        tmp_path,
        ["custom-model", "llm-small", f"../{outside.name}", "missing"],
        llm_catalog=LLM_CATALOG,
        embed_catalog=EMBED_CATALOG,
    )

    assert removed == ["custom-model"]
    assert set(failures) == {"llm-small", f"../{outside.name}", "missing"}
    assert catalog.is_dir()
    assert outside.is_dir()


def test_delete_non_catalog_model_dirs_unlinks_symlink_without_touching_target(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-linked-target"
    outside.mkdir()
    (outside / "model.gguf").write_bytes(b"model")
    link = tmp_path / "custom-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    removed, failures = delete_non_catalog_model_dirs(
        tmp_path, [link.name], llm_catalog=[], embed_catalog=[],
    )

    assert removed == ["custom-link"]
    assert failures == {}
    assert not link.exists()
    assert outside.is_dir()


def test_delete_non_catalog_model_dirs_reports_filesystem_failure(monkeypatch, tmp_path):
    target = tmp_path / "locked-model"
    target.mkdir()
    (target / "model.gguf").write_bytes(b"model")

    def fail_delete(_):
        raise PermissionError("locked")

    monkeypatch.setattr("model_inventory.shutil.rmtree", fail_delete)
    removed, failures = delete_non_catalog_model_dirs(
        tmp_path, [target.name], llm_catalog=[], embed_catalog=[],
    )

    assert removed == []
    assert failures == {"locked-model": "locked"}
    assert target.is_dir()


def test_delete_non_catalog_model_dirs_rejects_non_model_directory(tmp_path):
    target = tmp_path / "notes"
    target.mkdir()
    (target / "keep.txt").write_text("important")

    removed, failures = delete_non_catalog_model_dirs(
        tmp_path, [target.name], llm_catalog=[], embed_catalog=[],
    )

    assert removed == []
    assert failures == {"notes": "directory does not contain a GGUF model"}
    assert (target / "keep.txt").read_text() == "important"
