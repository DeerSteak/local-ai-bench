import json

import pytest

from scripts.release.model_catalog_inventory import (
    build_incumbent_inventory, load_incumbent_register,
)
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


def test_checked_in_incumbent_register_covers_the_exact_active_catalog():
    inventory = build_incumbent_inventory(load_incumbent_register())
    expected = len(LLM_MODELS) + len(EMBED_MODELS) + len(IMAGE_MODELS)
    assert len(inventory["incumbents"]) == expected == 19
    assert all(record["role"] for record in inventory["incumbents"])
    assert all(record["decision"] == "pending_evidence" for record in inventory["incumbents"])


def test_inventory_carries_exact_runtime_artifacts_and_image_access_state():
    records = {
        (record["family"], record["id"]): record
        for record in build_incumbent_inventory(load_incumbent_register())["incumbents"]
    }
    split = records[("llm", "qwen3-coder-next:80b-a3b-q4_K_M")]
    assert len(split["selected_artifacts"]["llamacpp"]["files"]) == 4
    assert split["selected_artifacts"]["vllm"]["repo"] == (
        "bullpoint/Qwen3-Coder-Next-AWQ-4bit"
    )
    assert records[("image", "sdxl")]["selected_artifacts"]["comfyui"]["gated"] is False
    assert records[("image", "flux2-dev")]["selected_artifacts"]["comfyui"]["gated"] is True


def test_register_rejects_duplicates_and_catalog_drift(tmp_path):
    path = tmp_path / "incumbents.json"
    record = {"id": "one", "family": "llm", "upstream": "owner/model", "role": "role"}
    path.write_text(json.dumps({"schema_version": 1, "incumbents": [record, record]}))
    with pytest.raises(ValueError, match="unique"):
        load_incumbent_register(path)

    current = load_incumbent_register()
    with pytest.raises(ValueError, match="mismatch"):
        build_incumbent_inventory(current[:-1])
