import json
from types import SimpleNamespace

import pytest

from scripts.release.model_catalog_audit import (
    DEFAULT_CANDIDATES, audit_pipeline_source, audit_repository, build_source_audit,
    load_candidate_register,
    source_status,
)
from scripts.release.model_catalog_inventory import load_incumbent_register
from scripts.setup.model_import import ImportVariant, RepositoryInspection


def info(*, repo="owner/model", sha="abc123", gated: bool | str = False, private=False,
         license="apache-2.0", license_name=None, base_model=None, files=(), tags=(),
         downloads=123, likes=4):
    return SimpleNamespace(
        id=repo, sha=sha, gated=gated, private=private,
        card_data=SimpleNamespace(
            license=license, license_name=license_name, base_model=base_model,
        ),
        pipeline_tag="text-generation",
        library_name="transformers",
        tags=list(tags),
        downloads=downloads, likes=likes,
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
    comparison = {"role": "role", "incumbents": ["incumbent"]}
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"schema_version": 2, "candidates": [
        {"id": "same", "family": "llm", **comparison,
         "sources": {"upstream": "a/b", "gguf": "c/d", "vllm": "e/f"}},
        {"id": "same", "family": "image", **comparison, "sources": {"upstream": "e/f"}},
    ]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 2, "candidates": [
        {"id": "embed", "family": "embedding", **comparison,
         "sources": {"upstream": "a/b"}},
    ]}))
    with pytest.raises(ValueError, match="GGUF"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 2, "candidates": [
        {"id": "image", "family": "image", **comparison,
         "sources": {"upstream": "a/b"}},
    ]}))
    with pytest.raises(ValueError, match="pipeline"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 2, "candidates": [{
        "id": "embed", "family": "embedding", **comparison,
        "gguf_provenance": "assumed",
        "sources": {"upstream": "a/b", "gguf": "a/b-GGUF"},
    }]}))
    with pytest.raises(ValueError, match="GGUF provenance"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 2, "candidates": [{
        "id": "model", "family": "llm",
        "sources": {"upstream": "a/b", "gguf": "c/d", "vllm": "e/f"},
    }]}))
    with pytest.raises(ValueError, match="measurable role"):
        load_candidate_register(path)

    path.write_text(json.dumps({"schema_version": 2, "candidates": [{
        "id": "model", "family": "llm", **comparison,
        "sources": {"upstream": "a/b", "gguf": "c/d"},
    }]}))
    with pytest.raises(ValueError, match="vLLM source"):
        load_candidate_register(path)


def test_candidate_comparisons_reference_current_incumbents_in_the_same_family():
    incumbents = {record["id"]: record for record in load_incumbent_register()}
    for candidate in load_candidate_register(DEFAULT_CANDIDATES):
        assert candidate["incumbents"]
        assert all(incumbent in incumbents for incumbent in candidate["incumbents"])
        assert all(incumbents[incumbent]["family"] == candidate["family"]
                   for incumbent in candidate["incumbents"])


def test_llm_candidates_use_distinct_vllm_sources_and_drop_superseded_nano():
    candidates = load_candidate_register(DEFAULT_CANDIDATES)
    by_id = {candidate["id"]: candidate for candidate in candidates}
    assert "nemotron-nano-9b-v2" not in by_id
    assert by_id["gemma-4-26b-a4b"]["sources"]["gguf"] == \
        "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert by_id["nemotron-3.5-lightning-30b-a3b"]["sources"] == {
        "upstream": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "gguf": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "vllm": "Local-Axiom-AI/Nemotron-3.5-Lightning-awq",
    }
    for candidate in candidates:
        if candidate["family"] == "llm":
            assert candidate["sources"]["vllm"] != candidate["sources"]["upstream"]


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
    assert record["downloads"] == 123
    assert record["likes"] == 4
    assert record["artifact"] == {
        "kind": "safetensors", "files": ["model.safetensors"],
        "support_files": ["config.json"], "size": 10,
    }


def test_repository_audit_normalizes_named_other_license():
    api = FakeApi({"owner/model": info(license="other", license_name="openmdw-1.1")})
    record = audit_repository(
        "owner/model", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda *_args: {},
    )
    assert record["license"] == "openmdw-1.1"


def test_repository_audit_uses_quantization_manifest_provenance():
    api = FakeApi({"owner/quant": info(files=(
        ("config.json", 1), ("quantization_manifest.json", 1),
    ))})
    values = {
        "config.json": {},
        "quantization_manifest.json": {
            "source_model": "owner/upstream", "source_revision": "source123",
        },
    }
    record = audit_repository(
        "owner/quant", "vllm", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda _repo, _revision, filename: values[filename],
    )
    assert record["base_models"] == ["owner/upstream"]
    assert record["quantization_provenance"] == {
        "source_model": "owner/upstream", "source_revision": "source123",
    }


def test_repository_audit_records_invalid_quantization_manifest_without_losing_config():
    api = FakeApi({"owner/quant": info(files=(
        ("config.json", 1), ("quantization_manifest.json", 1),
    ))})

    def read_json(_repo, _revision, filename):
        if filename == "quantization_manifest.json":
            raise ValueError("invalid manifest")
        return {"model_type": "model"}

    record = audit_repository(
        "owner/quant", "vllm", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo), read_json=read_json,
    )
    assert record["quantization_provenance_error"] == "ValueError"
    assert record["configuration"]["model_type"] == "model"


def test_repository_audit_does_not_accept_manifest_without_source_revision():
    api = FakeApi({"owner/quant": info(files=(("quantization_manifest.json", 1),))})
    record = audit_repository(
        "owner/quant", "vllm", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda *_args: {"source_model": "owner/upstream"},
    )
    assert record["base_models"] == []
    assert "quantization_provenance" not in record


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
    vllm = {
        "repo": "e/f", "revision": "ghi", "gated": False, "private": False,
        "license": "apache-2.0", "base_models": [], "artifact": None,
        "configuration": None,
    }
    status, reasons = source_status(
        {"id": "model", "family": "llm"},
        {"upstream": missing, "gguf": gguf, "vllm": vllm},
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
        "vLLM provenance does not identify the selected upstream repository",
        "vLLM artifact could not be resolved",
        "vLLM artifact is not a supported 4-bit quantization",
    ]


def test_source_status_accepts_explicit_exact_variant_from_same_publisher():
    candidate = {
        "id": "embed", "family": "embedding",
        "gguf_provenance": "publisher_exact_variant",
    }
    sources = {
        "upstream": {
            "repo": "Qwen/Qwen3-Embedding-0.6B", "private": False, "gated": False,
            "license": "apache-2.0", "artifact": {"files": ["model.safetensors"]},
            "configuration": {}, "custom_code": False,
        },
        "gguf": {
            "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF", "private": False,
            "gated": False, "license": "apache-2.0",
            "base_models": ["Qwen/Qwen3-0.6B-Base"],
            "artifact": {"files": ["model.gguf"]},
        },
    }
    assert source_status(candidate, sources) == ("source_ready", [])


def test_full_audit_preserves_candidate_order_and_derives_status():
    candidates = [{
        "id": "model", "family": "embedding", "name": "Model",
        "role": "role", "incumbents": ["incumbent"],
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
    assert result["schema_version"] == 2
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
        "quantization": None,
        "pipeline_class": None,
    }


@pytest.mark.parametrize(("quantization", "expected"), [
    ({"quant_method": "bitsandbytes", "load_in_4bit": True},
     {"method": "bitsandbytes", "bits": 4, "format": None}),
    ({"quant_method": "awq", "bits": 4, "checkpoint_format": "gemm"},
     {"method": "awq", "bits": 4, "format": "gemm"}),
    ({"quant_method": "compressed-tensors", "format": "pack-quantized",
      "config_groups": {"group": {"weights": {"num_bits": 4}}}},
     {"method": "compressed-tensors", "bits": 4, "format": "pack-quantized"}),
])
def test_configuration_metadata_normalizes_supported_four_bit_formats(quantization, expected):
    api = FakeApi({"owner/model": info(files=(("config.json", 1),))})
    record = audit_repository(
        "owner/model", "vllm", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
        read_json=lambda *_args: {"quantization_config": quantization},
    )
    assert record["configuration"]["quantization"] == expected


def test_llm_source_status_requires_q4_gguf_and_provenanced_four_bit_vllm():
    upstream = {
        "repo": "owner/model", "private": False, "gated": False,
        "license": "apache-2.0", "artifact": {"files": ["model.safetensors"]},
        "configuration": {"chat_template": "chat_template.jinja"},
        "custom_code": False,
    }
    gguf = {
        "repo": "owner/model-GGUF", "private": False, "gated": False,
        "license": "apache-2.0", "base_models": ["owner/model"],
        "artifact": {"label": "Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
    }
    vllm = {
        "repo": "quants/model-awq", "private": False, "gated": False,
        "license": "apache-2.0", "base_models": ["owner/model"],
        "artifact": {"files": ["model.safetensors"]},
        "configuration": {"quantization": {"method": "awq", "bits": 4}},
    }
    candidate = {"id": "model", "family": "llm"}
    assert source_status(candidate, {
        "upstream": upstream, "gguf": gguf, "vllm": vllm,
    }) == ("source_ready", [])

    upstream["license"] = "openmdw-1.1"
    gguf["license"] = "openmdw-1.1"
    vllm["license"] = "openmdw-1.1"
    assert source_status(candidate, {
        "upstream": upstream, "gguf": gguf, "vllm": vllm,
    }) == ("source_ready", [])

    gguf["artifact"]["label"] = "model-Q8_0.gguf"
    vllm["configuration"]["quantization"] = {"method": None, "bits": 4}
    vllm["base_models"] = []
    assert source_status(candidate, {
        "upstream": upstream, "gguf": gguf, "vllm": vllm,
    }) == ("blocked", [
        "GGUF artifact is not a 4-bit Q4 variant",
        "vLLM provenance does not identify the selected upstream repository",
        "vLLM artifact is not a supported 4-bit quantization",
    ])


def test_llm_source_status_rejects_mlx_quantization_metadata():
    sources = {
        "upstream": {
            "repo": "owner/model", "private": False, "gated": False,
            "license": "apache-2.0", "artifact": {"files": ["model.safetensors"]},
            "configuration": {"chat_template": "chat_template.jinja"},
            "custom_code": False,
        },
        "gguf": {
            "repo": "owner/model-GGUF", "private": False, "gated": False,
            "license": "apache-2.0", "base_models": ["owner/model"],
            "artifact": {"label": "model-Q4_K_M.gguf", "files": ["model.gguf"]},
        },
        "vllm": {
            "repo": "owner/model-MLX-4bit", "private": False, "gated": False,
            "license": "apache-2.0", "base_models": ["owner/model"],
            "artifact": {"files": ["model.safetensors"]},
            "configuration": {"quantization": {"method": None, "bits": 4}},
        },
    }
    assert source_status({"id": "model", "family": "llm"}, sources) == (
        "blocked", ["vLLM artifact is not a supported 4-bit quantization"],
    )


def test_source_status_keeps_unreviewed_named_license_blocked():
    sources = {
        "upstream": {
            "repo": "owner/model", "private": False, "gated": False,
            "license": "custom-1.0", "artifact": {"files": ["model.safetensors"]},
            "configuration": {"chat_template": "chat_template.jinja"},
            "custom_code": False,
        },
        "gguf": {
            "repo": "owner/model-GGUF", "private": False, "gated": False,
            "license": "custom-1.0", "base_models": ["owner/model"],
            "artifact": {"label": "model-Q4_K_M.gguf", "files": ["model.gguf"]},
        },
        "vllm": {
            "repo": "owner/model-awq", "private": False, "gated": False,
            "license": "custom-1.0", "base_models": ["owner/model"],
            "artifact": {"files": ["model.safetensors"]},
            "configuration": {"quantization": {"method": "awq", "bits": 4}},
        },
    }
    assert source_status({"id": "model", "family": "llm"}, sources) == (
        "blocked", ["upstream custom-1.0 license requires review"],
    )


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
