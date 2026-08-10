from types import SimpleNamespace

import pytest

from scripts.setup.model_import import (
    default_custom_tag, inspect_repository, normalize_hf_repo, valid_custom_tag,
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
        "model-00001-of-00002.safetensors": 30,
        "model-00002-of-00002.safetensors": 40,
        "adapter_model.safetensors": 3,
    }))

    assert inspection.revision == "commit"
    assert len(inspection.llama_variants) == 1
    assert inspection.llama_variants[0].size == 30
    assert inspection.vllm_variant is not None
    assert inspection.vllm_variant.size == 70


def test_inspection_reports_repo_with_no_engine_artifacts():
    inspection = inspect_repository("owner/tokenizer", api=FakeApi({"tokenizer.json": 5}))
    assert inspection.llama_variants == ()
    assert inspection.vllm_variant is None


def test_custom_tag_validation_and_default():
    assert default_custom_tag("owner/My_Model-GGUF") == "my_model-gguf"
    assert valid_custom_tag("my-model_1.0")
    assert not valid_custom_tag("owner/model")

