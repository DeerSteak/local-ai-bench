import json
from types import SimpleNamespace

import pytest

from scripts.release.model_catalog_audit import (
    audit_pipeline_source, audit_repository, build_source_audit, load_candidate_register,
    source_status,
)
from scripts.setup.model_import import ImportVariant, RepositoryInspection


def info(*, repo="owner/model", sha="abc123", gated: bool | str = False, private=False,
         license="apache-2.0", base_model=None, files=(), tags=()):
    return SimpleNamespace(
        id=repo, sha=sha, gated=gated, private=private,
        card_data=SimpleNamespace(license=license, base_model=base_model),
        pipeline_tag="text-generation",
        library_name="transformers",
        tags=list(tags),
        siblings=[SimpleNamespace(
            rfilename=name, size=size,
            lfs=SimpleNamespace(size=size, sha256=f"sha-{name}"),
        ) for name, size in files],
    )


class FakeApi:
    def __init__(self, values):
        self.values = values

    def model_info(self, repo, **_kwargs):
        return self.values[repo]


def inspection(repo, *, vllm=True, gguf=True):
    return RepositoryInspection(
        repo=repo, revision="abc123",
        llama_variants=(ImportVariant("q4", "model-Q4_K_M.gguf", ("model-Q4_K_M.gguf",), 12),)
        if gguf else (),
        vllm_variant=ImportVariant(
            "snapshot", "Safetensors repository snapshot", ("model.safetensors",), 10,
            ("config.json",),
        ) if vllm else None,
        gated=False,
    )


def test_candidate_register_rejects_duplicate_ids_and_missing_sources(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"schema_version": 1, "candidates": [
        {"id": "same", "family": "llm", "sources": {"upstream": "a/b", "gguf": "c/d"}},
        {"id": "same", "family": "image", "sources": {"upstream": "e/f"}},
    ]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 1, "candidates": [
        {"id": "embed", "family": "embedding", "sources": {"upstream": "a/b"}},
    ]}))
    with pytest.raises(ValueError, match="GGUF"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 1, "candidates": [
        {"id": "image", "family": "image", "sources": {"upstream": "a/b"}},
    ]}))
    with pytest.raises(ValueError, match="pipeline"):
        load_candidate_register(path)


def test_repository_audit_records_exact_revision_and_artifact_identity():
    api = FakeApi({"owner/model": info(files=(("model.safetensors", 10),))})
    record = audit_repository(
        "owner/model", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda *_args: {},
    )
    assert record["revision"] == "abc123"
    assert record["license"] == "apache-2.0"
    assert record["base_models"] == []
    assert record["artifact"] == {
        "kind": "safetensors", "files": ["model.safetensors"],
        "support_files": ["config.json"], "size": 10,
    }


def test_image_repository_records_pipeline_files():
    model_info = info(files=(("transformer/model.safetensors", 99),))
    api = FakeApi({"owner/image": model_info})
    record = audit_repository(
        "owner/image", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo, vllm=False, gguf=False),
        read_json=lambda *_args: {},
    )
    assert record["artifact"]["kind"] == "pipeline"


def test_pipeline_source_requires_every_exact_file_with_size_and_digest():
    api = FakeApi({"owner/pipeline": info(
        repo="owner/pipeline", files=(("model.safetensors", 99),),
    )})
    record = audit_pipeline_source({
        "repo": "owner/pipeline", "files": ["model.safetensors", "missing.safetensors"],
    }, api=api)
    assert record["files"] == [
        {"name": "model.safetensors", "size": 99, "sha256": "sha-model.safetensors"},
        {"name": "missing.safetensors", "size": None, "sha256": None},
    ]
    status, reasons = source_status(
        {"id": "image", "family": "image"},
        {"upstream": {
            "repo": "owner/image", "private": False, "gated": False,
            "license": "apache-2.0", "artifact": {"kind": "pipeline"},
            "configuration": {},
        }, "pipeline": [record]},
    )
    assert status == "blocked"
    assert reasons == ["pipeline artifact is unresolved: owner/pipeline"]


def test_source_status_surfaces_access_license_and_artifact_gates():
    missing = {
        "repo": "a/b", "revision": "abc", "gated": "manual", "private": False,
        "license": None, "base_models": [], "artifact": None, "configuration": None,
    }
    gguf = {
        "repo": "c/d", "revision": "def", "gated": False, "private": True,
        "license": "apache-2.0", "base_models": [], "artifact": None,
    }
    status, reasons = source_status(
        {"id": "model", "family": "llm"}, {"upstream": missing, "gguf": gguf},
    )
    assert status == "blocked"
    assert reasons == [
        "upstream repository requires access approval",
        "upstream license is not declared",
        "upstream artifact could not be resolved",
        "upstream configuration could not be inspected",
        "GGUF repository is not publicly accessible",
        "GGUF provenance does not identify the selected upstream repository",
        "GGUF artifact could not be resolved",
    ]


def test_full_audit_preserves_candidate_order_and_derives_status():
    candidates = [{
        "id": "model", "family": "embedding", "name": "Model",
        "sources": {"upstream": "owner/model", "gguf": "owner/model-gguf"},
    }]
    api = FakeApi({
        "owner/model": info(repo="owner/model"),
        "owner/model-gguf": info(repo="owner/model-gguf", base_model="owner/model"),
    })
    result = build_source_audit(
        candidates, api=api, inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda *_args: {},
    )
    assert result["schema_version"] == 1
    assert result["candidates"][0]["id"] == "model"
    assert result["candidates"][0]["status"] == "source_ready"


def test_configuration_metadata_records_context_template_and_publisher_sampling():
    files = (
        ("config.json", 1), ("tokenizer_config.json", 1),
        ("generation_config.json", 1), ("chat_template.jinja", 1),
    )
    api = FakeApi({"owner/model": info(files=files, tags=("custom_code",))})
    values = {
        "config.json": {
            "model_type": "moe", "architectures": ["ModelForCausalLM"],
            "max_position_embeddings": 131072, "torch_dtype": "bfloat16",
            "hidden_size": 4096, "num_hidden_layers": 40,
            "num_experts": 128, "num_experts_per_tok": 4,
        },
        "tokenizer_config.json": {"chat_template": "ignored because standalone wins"},
        "generation_config.json": {"temperature": 0.7, "top_p": 0.9, "bos_token_id": 1},
    }
    record = audit_repository(
        "owner/model", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda _repo, _revision, filename: values[filename],
    )
    assert record["custom_code"] is True
    assert record["configuration"] == {
        "model_type": "moe", "architectures": ["ModelForCausalLM"],
        "context_tokens": 131072, "dtype": "bfloat16", "hidden_size": 4096,
        "num_hidden_layers": 40, "num_experts": 128, "num_experts_per_token": 4,
        "chat_template": "chat_template.jinja",
        "publisher_sampling": {"temperature": 0.7, "top_p": 0.9},
        "pipeline_class": None,
    }


def test_unreadable_gated_configuration_is_recorded_without_losing_source_identity():
    api = FakeApi({"owner/model": info(files=(("config.json", 1),), gated="manual")})

    def denied(*_args):
        raise PermissionError("token omitted")

    record = audit_repository(
        "owner/model", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo), read_json=denied,
    )
    assert record["revision"] == "abc123"
    assert record["configuration"] is None
    assert record["configuration_error"] == "PermissionError"
