import json

import pytest

from scripts.runtime.image_model_spec import (
    load_audit_image_model, select_audit_image_model, validate_audit_image_model,
)


def model():
    return {
        "audit_candidate": True, "artifact_digest": "a" * 64,
        "label": "Z-Image Turbo", "short": "z-image-turbo", "tier": "medium",
        "checkpoint": "z_image_turbo_bf16.safetensors",
        "checkpoint_folder": "diffusion_models", "workflow": "z_image",
        "steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple",
    }


def test_audit_image_model_loads_exact_schema_and_selects_only_image_workload(tmp_path):
    path = tmp_path / "image.json"
    path.write_text(json.dumps({"schema_version": 1, "model": model()}), encoding="utf-8")
    loaded = load_audit_image_model(path)
    assert loaded == model()
    assert select_audit_image_model(["img"], ["z-image-turbo"], loaded) == [loaded]
    with pytest.raises(ValueError, match="requires --tests img"):
        select_audit_image_model(["llm"], None, loaded)
    with pytest.raises(ValueError, match="conflicts"):
        select_audit_image_model(["img"], ["sdxl"], loaded)


def test_audit_image_model_rejects_unknown_fields_workflows_and_sampling_values():
    invalid = {**model(), "extra": True}
    with pytest.raises(ValueError, match="invalid fields"):
        validate_audit_image_model(invalid)
    invalid = {**model(), "workflow": "flux2"}
    with pytest.raises(ValueError, match="not a supported"):
        validate_audit_image_model(invalid)
    invalid = {**model(), "cfg": True}
    with pytest.raises(ValueError, match="sampling settings"):
        validate_audit_image_model(invalid)
