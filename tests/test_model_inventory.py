from scripts.setup.model_inventory import (
    build_model_inventory,
    classify_engine_models,
    delete_non_catalog_model_dirs,
    engine_download_size,
    engine_fit_report,
    engine_fit_warnings,
    fits_any_engine,
    format_engine_sizes,
    engine_model_complete,
    engine_model_dir,
    models_missing_engine_support,
    find_non_catalog_model_dirs,
    format_model_inventory,
    installed_image_models,
    model_tag_slug,
    sanitize_tag_to_short,
)
from scripts.setup.custom_models import load_custom_models, save_custom_model


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
    {"short": "image-three", "label": "Image Three", "checkpoint": "three.safetensors",
     "checkpoint_folder": "diffusion_models"},
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


def test_custom_model_inventory_preserves_registered_display_label():
    inventory = classify_engine_models(
        [{"tag": "custom", "label": "Friendly", "size": 10}], [], [],
    )
    assert inventory["custom"][0]["label"] == "Friendly"


def test_installed_images_use_explicit_comfyui_path(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "two.safetensors"
    checkpoint.write_bytes(b"12345")

    installed = installed_image_models(tmp_path, IMAGE_CATALOG)

    assert [model["short"] for model in installed] == ["image-two"]
    assert installed[0]["path"] == checkpoint
    assert installed[0]["size"] == 5


def test_installed_images_resolve_catalog_checkpoint_folder(tmp_path):
    diffusion_models = tmp_path / "diffusion_models"
    diffusion_models.mkdir(parents=True)
    checkpoint = diffusion_models / "three.safetensors"
    checkpoint.write_bytes(b"z-image")

    installed = installed_image_models(tmp_path, IMAGE_CATALOG)

    assert [model["short"] for model in installed] == ["image-three"]
    assert installed[0]["path"] == checkpoint
    assert installed[0]["size"] == 7


def test_installed_image_requires_every_support_asset(tmp_path):
    model = {
        "short": "pipeline", "label": "Pipeline", "checkpoint": "model.safetensors",
        "checkpoint_folder": "diffusion_models",
        "support_assets": [
            {"folder": "text_encoders", "name": "encoder.safetensors"},
            {"folder": "vae", "name": "vae.safetensors"},
        ],
    }
    for folder, name in (
        ("diffusion_models", "model.safetensors"),
        ("text_encoders", "encoder.safetensors"),
    ):
        path = tmp_path / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")

    assert installed_image_models(tmp_path, [model]) == []
    vae = tmp_path / "vae" / "vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"asset")
    assert [entry["short"] for entry in installed_image_models(tmp_path, [model])] == [
        "pipeline",
    ]


def test_installed_images_empty_when_checkpoint_directory_missing(tmp_path):
    assert installed_image_models(tmp_path, IMAGE_CATALOG) == []


def test_build_inventory_reads_engine_once_and_adds_images(monkeypatch, tmp_path):
    engine = FakeEngine([{"tag": LLM_CATALOG[0]["tag"], "size": 3}])
    monkeypatch.setattr("scripts.setup.model_inventory.LLM_MODELS", LLM_CATALOG)
    monkeypatch.setattr("scripts.setup.model_inventory.EMBED_MODELS", EMBED_CATALOG)
    monkeypatch.setattr("scripts.setup.model_inventory.IMAGE_MODELS", IMAGE_CATALOG)
    checkpoints = tmp_path / "checkpoints"
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


def test_deleting_imported_llamacpp_folder_forgets_its_registry_entry(tmp_path):
    registry = tmp_path / "custom.json"
    target = tmp_path / "custom-model"
    target.mkdir()
    (target / "model.gguf").write_bytes(b"model")
    save_custom_model({"engine": "llamacpp", "tag": target.name}, registry)

    removed, failures = delete_non_catalog_model_dirs(
        tmp_path, [target.name], llm_catalog=[], embed_catalog=[], registry_path=registry,
    )

    assert removed == [target.name] and failures == {}
    assert load_custom_models(registry) == []


def test_delete_non_catalog_model_dirs_unlinks_symlink_without_touching_target(
        tmp_path, symlink_or_skip):
    outside = tmp_path.parent / f"{tmp_path.name}-linked-target"
    outside.mkdir()
    (outside / "model.gguf").write_bytes(b"model")
    link = symlink_or_skip(tmp_path / "custom-link", outside, directory=True)

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

    monkeypatch.setattr("scripts.setup.model_inventory.shutil.rmtree", fail_delete)
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


# ── per-engine model storage ──

def test_engine_model_dir_namespaces_by_engine(tmp_path):
    assert engine_model_dir(tmp_path, "llamacpp", "qwen3.5:9b-q4_K_M") == tmp_path / "llamacpp" / "qwen3.5_9b-q4_K_M"
    assert engine_model_dir(tmp_path, "vllm", "qwen3.5:9b-q4_K_M") == tmp_path / "vllm" / "qwen3.5_9b-q4_K_M"


def test_engine_download_size_prefers_the_engines_own_weights():
    model = {"download_size": "~5.5 GB", "vllm_download_size": "~12.4 GB"}
    assert engine_download_size(model, "llamacpp") == "~5.5 GB"
    assert engine_download_size(model, "vllm") == "~12.4 GB"


def test_engine_download_size_is_unknown_when_no_vllm_size_is_listed():
    """A missing vllm_download_size must report unknown, not the GGUF's own size —
    vLLM downloads a different weight format so the two sizes aren't interchangeable."""
    assert engine_download_size({"download_size": "~5.5 GB"}, "vllm") is None


def test_llamacpp_completeness_needs_every_listed_gguf(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    files = ["a-00001-of-00002.gguf", "a-00002-of-00002.gguf"]
    (model_dir / files[0]).touch()
    assert engine_model_complete(model_dir, "llamacpp", files) is False
    (model_dir / files[1]).touch()
    assert engine_model_complete(model_dir, "llamacpp", files) is True


def test_completeness_is_false_for_a_missing_directory(tmp_path):
    assert engine_model_complete(tmp_path / "nope", "llamacpp", ["a.gguf"]) is False


def test_models_missing_engine_support_only_applies_to_vllm():
    models = [{"tag": "a", "vllm_repo": "org/a"}, {"tag": "b"}]
    assert models_missing_engine_support(models, "vllm") == ["b"]
    assert models_missing_engine_support(models, "llamacpp") == []


def test_every_catalog_model_has_vllm_weights_defined():
    from scripts.workloads.models import EMBED_MODELS, LLM_MODELS
    assert models_missing_engine_support(LLM_MODELS + EMBED_MODELS, "vllm") == []
    for model in LLM_MODELS + EMBED_MODELS:
        assert "/" in model["vllm_repo"], model["tag"]
        assert model["vllm_download_size"].startswith("~")


# ── per-engine size and memory fit ──

MIXED = {"tag": "m", "download_size": "~6.2 GB", "vllm_download_size": "~12.4 GB",
         "vllm_repo": "org/m-awq"}


def test_fit_report_uses_each_engines_own_weights():
    report = engine_fit_report(MIXED, ["llamacpp", "vllm"], ceiling_gb=100)
    assert report["llamacpp"]["size"] == "~6.2 GB"
    assert report["vllm"]["size"] == "~12.4 GB"
    assert report["vllm"]["needed_gb"] > report["llamacpp"]["needed_gb"]


def test_a_model_can_fit_one_engine_and_not_the_other():
    report = engine_fit_report(MIXED, ["llamacpp", "vllm"], ceiling_gb=12.0)
    assert report["llamacpp"]["fits"] is True
    assert report["vllm"]["fits"] is False
    assert fits_any_engine(report) is True, "still worth downloading for llama.cpp"


def test_vllm_only_selection_reports_only_vllm():
    report = engine_fit_report(MIXED, ["vllm"], ceiling_gb=12.0)
    assert list(report) == ["vllm"]
    assert fits_any_engine(report) is False
    assert format_engine_sizes(report) == "~12.4 GB"


def test_llamacpp_only_selection_is_unchanged():
    report = engine_fit_report(MIXED, ["llamacpp"], ceiling_gb=12.0)
    assert list(report) == ["llamacpp"]
    assert format_engine_sizes(report) == "~6.2 GB"
    assert engine_fit_warnings(report, 12.0) == []


def test_both_engines_label_names_each_one():
    report = engine_fit_report(MIXED, ["llamacpp", "vllm"], ceiling_gb=12.0)
    assert format_engine_sizes(report) == "llama.cpp ~6.2 GB · vLLM ~12.4 GB"


def test_warnings_name_the_engine_only_when_several_are_selected():
    both = engine_fit_warnings(engine_fit_report(MIXED, ["llamacpp", "vllm"], 12.0), 12.0)
    assert both == ["vLLM needs ~14.9 GB, ~12.0 GB available"]
    single = engine_fit_warnings(engine_fit_report(MIXED, ["vllm"], 12.0), 12.0)
    assert single == ["needs ~14.9 GB, ~12.0 GB available"]


def test_unknown_ceiling_yields_no_verdict_and_no_warnings():
    report = engine_fit_report(MIXED, ["llamacpp", "vllm"], ceiling_gb=None)
    assert fits_any_engine(report) is None
    assert engine_fit_warnings(report, None) == []


def test_a_model_fitting_nothing_is_reported_as_unfit():
    report = engine_fit_report(MIXED, ["llamacpp", "vllm"], ceiling_gb=1.0)
    assert fits_any_engine(report) is False
    assert len(engine_fit_warnings(report, 1.0)) == 2


def test_a_model_without_vllm_weights_is_skipped_in_the_report():
    model = {"tag": "m", "download_size": "~6.2 GB"}
    report = engine_fit_report(model, ["llamacpp", "vllm"], ceiling_gb=100)
    assert list(report) == ["llamacpp"]
    assert format_engine_sizes(report) == "~6.2 GB"


def test_empty_engine_selection_has_no_verdict():
    assert fits_any_engine(engine_fit_report(MIXED, [], ceiling_gb=12.0)) is None


def test_real_catalog_sizes_differ_between_engines():
    from scripts.workloads.models import LLM_MODELS
    differing = [m for m in LLM_MODELS
                 if engine_download_size(m, "vllm") != engine_download_size(m, "llamacpp")]
    assert len(differing) == len(LLM_MODELS), "every LLM should carry its own vLLM size"


def test_image_checkpoints_have_no_engine_download_size():
    image = {"label": "SDXL", "checkpoint": "sd_xl_base_1.0.safetensors", "short": "sdxl"}
    assert engine_download_size(image, "llamacpp") is None
    assert engine_download_size(image, "vllm") is None
    assert engine_fit_report(image, ["llamacpp", "vllm"], 100) == {}
    assert fits_any_engine(engine_fit_report(image, ["llamacpp"], 100)) is None


def test_fit_report_skips_engines_without_a_size_but_keeps_the_others():
    partial = {"tag": "m", "vllm_download_size": "~9 GB", "vllm_repo": "org/m"}
    report = engine_fit_report(partial, ["llamacpp", "vllm"], 100)
    assert list(report) == ["vllm"]
