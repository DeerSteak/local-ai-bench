from types import SimpleNamespace

from scripts.release.model_catalog_incumbent_audit import (
    build_incumbent_source_audit, incumbent_source_status,
    selected_file_records, snapshot_artifact,
)


class Api:
    def model_info(self, repo, **_kwargs):
        files = [
            SimpleNamespace(rfilename="config.json", size=10, lfs=None),
            SimpleNamespace(rfilename="model.safetensors", size=20, lfs=None),
            SimpleNamespace(rfilename="model.gguf", size=30, lfs=None),
        ]
        return SimpleNamespace(
            sha=f"revision-{repo}", private=False, gated=False,
            card_data=SimpleNamespace(
                license="apache-2.0",
                base_model=["owner/upstream"] if repo != "owner/upstream" else [],
            ),
            siblings=files,
        )


def read_json(_repo, _revision, name):
    return {"model_type": "test", "max_position_embeddings": 4096} if name == "config.json" else {}


def test_incumbent_audit_pins_upstream_and_every_selected_runtime_source():
    inventory = {"incumbents": [{
        "id": "model", "family": "llm", "upstream": "owner/upstream", "role": "role",
        "label": "Model", "short": "model", "tier": "small", "params_b": 1,
        "decision": "pending_evidence",
        "selected_artifacts": {
            "llamacpp": {"repo": "owner/gguf", "files": ["model.gguf"]},
            "vllm": {"repo": "owner/vllm"},
        },
    }]}
    record = build_incumbent_source_audit(
        inventory, api=Api(), read_json=read_json,
    )["incumbents"][0]
    assert record["sources"]["upstream"]["revision"] == "revision-owner/upstream"
    assert record["sources"]["llamacpp"]["artifact"]["files"][0]["size"] == 30
    assert record["sources"]["vllm"]["artifact"] == {
        "weight_file_count": 1, "weight_size": 20, "has_config": True,
    }
    assert record["source_status"] == "source_ready"


def test_selected_artifact_and_snapshot_report_missing_evidence():
    repository = {"files": [{"name": "config.json", "size": 1, "sha256": None}]}
    assert selected_file_records(repository, ["missing.gguf"])[0]["size"] is None
    assert snapshot_artifact(repository) == {
        "weight_file_count": 0, "weight_size": 0, "has_config": True,
    }


def test_same_repository_snapshot_needs_no_redundant_base_model_provenance():
    record = {"sources": {
        "upstream": {
            "repo": "owner/model", "private": False, "gated": False,
            "license": "apache-2.0", "configuration": {},
        },
        "vllm": {
            "repo": "owner/model", "private": False, "gated": False,
            "license": "apache-2.0", "base_models": [],
            "artifact": {"weight_file_count": 1, "has_config": True},
        },
    }}
    assert incumbent_source_status(record) == ("source_ready", [])


def test_source_status_accumulates_access_license_configuration_and_artifact_gaps():
    record = {"sources": {
        "upstream": {
            "repo": "owner/upstream",
            "private": False, "gated": "manual", "license": "other",
            "configuration": None,
        },
        "llamacpp": {
            "repo": "owner/gguf", "private": False, "gated": False,
            "license": "apache-2.0",
            "base_models": ["owner/upstream"],
            "artifact": {"files": [{"name": "model.gguf", "size": None}]},
        },
        "vllm": {
            "repo": "owner/vllm", "private": False, "gated": False,
            "license": "apache-2.0",
            "base_models": ["owner/upstream"],
            "artifact": {"weight_file_count": 0, "has_config": False},
        },
    }}
    status, reasons = incumbent_source_status(record)
    assert status == "review_required"
    assert reasons == [
        "upstream repository requires access approval",
        "upstream other license requires review",
        "upstream configuration could not be inspected",
        "selected llamacpp artifact is unresolved",
        "selected vllm snapshot is incomplete",
    ]
