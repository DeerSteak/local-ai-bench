import json
from types import SimpleNamespace

import pytest

from scripts.release.model_catalog_audit import (
    audit_repository, build_source_audit, load_candidate_register, source_status,
)
from scripts.setup.model_import import ImportVariant, RepositoryInspection


def info(*, repo="owner/model", sha="abc123", gated=False, private=False,
         license="apache-2.0", base_model=None, files=()):
    return SimpleNamespace(
        id=repo, sha=sha, gated=gated, private=private,
        card_data=SimpleNamespace(license=license, base_model=base_model),
        pipeline_tag="text-generation",
        library_name="transformers",
        siblings=[SimpleNamespace(rfilename=name, size=size, lfs=None) for name, size in files],
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


def test_repository_audit_records_exact_revision_and_artifact_identity():
    api = FakeApi({"owner/model": info(files=(("model.safetensors", 10),))})
    record = audit_repository(
        "owner/model", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo),
    )
    assert record["revision"] == "abc123"
    assert record["license"] == "apache-2.0"
    assert record["base_models"] == []
    assert record["artifact"] == {
        "kind": "safetensors", "files": ["model.safetensors"],
        "support_files": ["config.json"], "size": 10,
    }


def test_image_repository_records_pipeline_files_but_stays_blocked_for_workflow_selection():
    model_info = info(files=(("transformer/model.safetensors", 99),))
    api = FakeApi({"owner/image": model_info})
    record = audit_repository(
        "owner/image", "upstream", api=api,
        inspect_fn=lambda repo, **kwargs: inspection(repo, vllm=False, gguf=False),
    )
    candidate = {"id": "image", "family": "image"}
    status, reasons = source_status(candidate, {"upstream": record})
    assert record["artifact"]["kind"] == "pipeline"
    assert status == "blocked"
    assert reasons == ["complete ComfyUI pipeline artifact selection remains pending"]


def test_source_status_surfaces_access_license_and_artifact_gates():
    missing = {
        "repo": "a/b", "revision": "abc", "gated": "manual", "private": False,
        "license": None, "base_models": [], "artifact": None,
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
        "GGUF repository is not publicly accessible",
        "GGUF provenance does not identify the selected upstream repository",
        "GGUF artifact could not be resolved",
    ]


def test_full_audit_preserves_candidate_order_and_derives_status():
    candidates = [{
        "id": "model", "family": "llm", "name": "Model",
        "sources": {"upstream": "owner/model", "gguf": "owner/model-gguf"},
    }]
    api = FakeApi({
        "owner/model": info(repo="owner/model"),
        "owner/model-gguf": info(repo="owner/model-gguf", base_model="owner/model"),
    })
    result = build_source_audit(
        candidates, api=api, inspect_fn=lambda repo, **kwargs: inspection(repo),
    )
    assert result["schema_version"] == 1
    assert result["candidates"][0]["id"] == "model"
    assert result["candidates"][0]["status"] == "source_ready"
