from types import SimpleNamespace

import pytest

from scripts.app import model_import_dialog
from scripts.app.model_import_dialog import (
    import_destination, import_validation_reason, import_variants,
    supported_import_engines,
)
from scripts.runtime import config
from scripts.setup.model_import import (
    ImportVariant, default_custom_tag, inspect_repository, normalize_hf_repo,
    preferred_variant, valid_custom_tag,
)


class FakeApi:
    def __init__(self, files, *, sha="commit", gated=False):
        self.files = files
        self.sha = sha
        self.gated = gated

    def model_info(self, repo, **_kwargs):
        siblings = [SimpleNamespace(rfilename=name, size=size, lfs=None)
                    for name, size in self.files.items()]
        return SimpleNamespace(siblings=siblings, sha=self.sha, gated=self.gated)


@pytest.mark.parametrize("value", [
    "owner/model", "https://huggingface.co/owner/model",
    "https://www.huggingface.co/owner/model/tree/main",
])
def test_normalize_hf_repo_accepts_ids_and_urls(value):
    assert normalize_hf_repo(value) == "owner/model"


@pytest.mark.parametrize("value", ["model", "http://huggingface.co/a/b", "https://example.com/a/b"])
def test_normalize_hf_repo_rejects_unsafe_or_incomplete_values(value):
    with pytest.raises(ValueError):
        normalize_hf_repo(value)


def test_inspection_groups_complete_gguf_parts_and_detects_vllm_snapshot():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "model-Q4_K_M-00001-of-00002.gguf": 10,
        "model-Q4_K_M-00002-of-00002.gguf": 20,
        "broken-00001-of-00002.gguf": 5,
        "mmproj-model.gguf": 2,
        "config.json": 1,
        "model.safetensors.index.json": 1,
        "model-00001-of-00002.safetensors": 30,
        "model-00002-of-00002.safetensors": 40,
        "unused.safetensors": 50,
        "adapter_model.safetensors": 3,
        "tokenizer.json": 4,
        "training_args.json": 5,
        "README.md": 6,
    }), read_repo_json=lambda _name: {"weight_map": {
        "layer.0": "model-00001-of-00002.safetensors",
        "layer.1": "model-00002-of-00002.safetensors",
    }})

    assert inspection.revision == "commit"
    assert len(inspection.llama_variants) == 1
    assert inspection.llama_variants[0].size == 30
    assert inspection.vllm_variant is not None
    assert inspection.vllm_variant.size == 70
    assert inspection.vllm_variant.files == (
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    )
    assert inspection.vllm_variant.support_files == (
        "config.json", "model.safetensors.index.json", "tokenizer.json",
    )


@pytest.mark.parametrize("index_data", [
    {"weight_map": {"layer": "missing.safetensors"}},
    {"weight_map": {"layer": "weights.bin"}},
    {"metadata": {}},
])
def test_inspection_rejects_invalid_safetensors_index(index_data):
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors.index.json": 1,
        "model-00001-of-00002.safetensors": 10,
    }), read_repo_json=lambda _name: index_data)

    assert inspection.vllm_variant is None


def test_inspection_rejects_multiple_unindexed_safetensors_files():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "first.safetensors": 10, "second.safetensors": 20,
    }))

    assert inspection.vllm_variant is None


def test_inspection_reports_repo_with_no_engine_artifacts():
    inspection = inspect_repository("owner/tokenizer", api=FakeApi({"tokenizer.json": 5}))
    assert inspection.llama_variants == ()
    assert inspection.vllm_variant is None


def test_custom_tag_validation_and_default():
    assert default_custom_tag("owner/My_Model-GGUF") == "my_model-gguf"
    assert valid_custom_tag("my-model_1.0")
    assert not valid_custom_tag("owner/model")


def test_preferred_variant_chooses_standard_q4_before_smaller_quants():
    variants = (
        ImportVariant("q2", "model-Q2_K.gguf", ("q2.gguf",), 2),
        ImportVariant("q4xl", "model-Q4_K_XL.gguf", ("q4xl.gguf",), 4),
        ImportVariant("q4m", "model-Q4_K_M.gguf", ("q4m.gguf",), 4),
    )
    assert preferred_variant(variants) == variants[2]
    assert preferred_variant(()) is None


def test_multipart_gguf_variants_keep_their_directories_separate():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "Q4_K_M/model-00001-of-00002.gguf": 1,
        "Q4_K_M/model-00002-of-00002.gguf": 2,
        "Q8_0/model-00001-of-00002.gguf": 3,
        "Q8_0/model-00002-of-00002.gguf": 4,
    }))

    assert [variant.files for variant in inspection.llama_variants] == [
        ("Q4_K_M/model-00001-of-00002.gguf", "Q4_K_M/model-00002-of-00002.gguf"),
        ("Q8_0/model-00001-of-00002.gguf", "Q8_0/model-00002-of-00002.gguf"),
    ]


def test_single_gguf_labels_include_the_disambiguating_path():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "Q4_K_M/model.gguf": 1, "Q8_0/model.gguf": 1,
    }))

    assert [variant.label for variant in inspection.llama_variants] == [
        "Q4_K_M/model.gguf", "Q8_0/model.gguf",
    ]


def test_auxiliary_filter_does_not_drop_model_names_containing_draft():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "redraft-model-Q4_K_M.gguf": 1, "draft-helper.gguf": 1,
        "model-mmproj-f16.gguf": 2, "model.dflash.Q8.gguf": 3,
    }))

    assert [variant.label for variant in inspection.llama_variants] == [
        "redraft-model-Q4_K_M.gguf",
    ]


def test_vllm_rejects_nested_weights_ignored_by_the_downloader():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "original/consolidated.safetensors": 10,
    }))

    assert inspection.vllm_variant is None


def test_vllm_rejects_index_references_to_nested_weights():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors.index.json": 1,
        "original/model.safetensors": 10,
    }), read_repo_json=lambda _name: {
        "weight_map": {"layer": "original/model.safetensors"},
    })

    assert inspection.vllm_variant is None


def test_vllm_prefers_canonical_index_when_multiple_are_present():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors.index.json": 1,
        "consolidated.safetensors.index.json": 1, "model.safetensors": 10,
    }), read_repo_json=lambda name: {
        "weight_map": {"layer": "model.safetensors"}
    } if name == "model.safetensors.index.json" else {})

    assert inspection.vllm_variant is not None
    assert inspection.vllm_variant.files == ("model.safetensors",)


def test_import_destination_requires_a_supported_engine(tmp_path):
    assert import_destination("llamacpp", "custom") == config.MODELS_DIR / "llamacpp" / "custom"
    assert import_destination("vllm", "custom", tmp_path) == tmp_path
    assert import_destination("", "custom") is None


def test_import_variants_requires_an_explicit_supported_engine():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors": 10, "model.gguf": 11,
    }))

    assert import_variants(inspection, "llamacpp") == inspection.llama_variants
    assert import_variants(inspection, "vllm") == (inspection.vllm_variant,)
    assert import_variants(inspection, "") == ()


def test_supported_import_engines_respects_artifacts_and_installation():
    inspection = inspect_repository("owner/model", api=FakeApi({
        "config.json": 1, "model.safetensors": 10, "model.gguf": 11,
    }))
    assert supported_import_engines(inspection, ["vllm", "llamacpp"]) == ["llamacpp", "vllm"]
    assert supported_import_engines(inspection, ["vllm"]) == ["vllm"]


def test_import_validation_rejects_stale_inspection_before_selection(tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 11}))
    reason = import_validation_reason(
        busy=False, inspection=inspection, inspected_request=("owner/model", "old"),
        current_request=("owner/model", "main"), engine="", engines=["llamacpp"],
        variant=None, tag="", label="", acknowledged=False,
    )
    assert reason == "Repository or revision changed; inspect it again."


def test_import_validation_accepts_available_selection(monkeypatch, tmp_path):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 11}))
    monkeypatch.setattr(model_import_dialog, "custom_model", lambda *_args: None)
    monkeypatch.setattr(model_import_dialog, "import_destination", lambda *_args: tmp_path)
    monkeypatch.setattr(model_import_dialog, "enough_disk_space", lambda *_args: True)
    reason = import_validation_reason(
        busy=False, inspection=inspection,
        inspected_request=("owner/model", inspection.revision),
        current_request=("owner/model", inspection.revision), engine="llamacpp",
        engines=["llamacpp"], variant=inspection.llama_variants[0], tag="custom-model",
        label="Custom Model", acknowledged=True,
    )
    assert reason == ""


@pytest.mark.parametrize(("changes", "expected"), [
    ({"busy": True}, "Operation in progress."),
    ({"inspection": None}, "Inspect a repository before importing."),
    ({"engine": "vllm"}, "Select an available engine and artifact variant."),
    ({"tag": "bad/tag"}, "Tag must use only letters, numbers, dots, underscores, or hyphens."),
    ({"label": " "}, "Display name is required."),
    ({"acknowledged": False}, "Acknowledge that runtime compatibility is unverified."),
])
def test_import_validation_rejects_invalid_states(monkeypatch, tmp_path, changes, expected):
    inspection = inspect_repository("owner/model", api=FakeApi({"model.gguf": 11}))
    monkeypatch.setattr(model_import_dialog, "custom_model", lambda *_args: None)
    monkeypatch.setattr(model_import_dialog, "import_destination", lambda *_args: tmp_path)
    monkeypatch.setattr(model_import_dialog, "enough_disk_space", lambda *_args: True)
    values = {
        "busy": False, "inspection": inspection,
        "inspected_request": ("owner/model", inspection.revision),
        "current_request": ("owner/model", inspection.revision), "engine": "llamacpp",
        "engines": ["llamacpp"], "variant": inspection.llama_variants[0],
        "tag": "custom-model", "label": "Custom Model", "acknowledged": True,
    }
    values.update(changes)
    assert import_validation_reason(**values) == expected
