"""Validated internal image-model specifications for catalog compatibility audits."""

import json
from pathlib import Path


AUDIT_IMAGE_SPEC_SCHEMA_VERSION = 1
AUDIT_IMAGE_MODEL_FIELDS = {
    "audit_candidate", "artifact_digest", "label", "short", "tier", "checkpoint",
    "checkpoint_folder", "workflow", "steps", "cfg", "sampler", "scheduler",
}


def validate_audit_image_model(model) -> dict:
    if not isinstance(model, dict) or set(model) != AUDIT_IMAGE_MODEL_FIELDS:
        raise ValueError("audit image-model specification has invalid fields")
    if model["audit_candidate"] is not True or model["workflow"] != "z_image":
        raise ValueError("audit image-model specification is not a supported candidate")
    for key in ("artifact_digest", "label", "short", "tier", "checkpoint",
                "checkpoint_folder", "sampler", "scheduler"):
        if not isinstance(model[key], str) or not model[key]:
            raise ValueError(f"audit image-model specification has invalid {key}")
    if not isinstance(model["steps"], int) or model["steps"] < 1 \
            or isinstance(model["cfg"], bool) or not isinstance(model["cfg"], (int, float)) \
            or model["cfg"] < 0:
        raise ValueError("audit image-model specification has invalid sampling settings")
    return dict(model)


def load_audit_image_model(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != AUDIT_IMAGE_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported audit image-model specification")
    return validate_audit_image_model(value.get("model"))


def select_audit_image_model(tests, image_selectors, model) -> list[dict]:
    candidate = validate_audit_image_model(model)
    if "img" not in tests:
        raise ValueError("--audit-image-model requires --tests img")
    if image_selectors and image_selectors != [candidate["short"]]:
        raise ValueError("--image-models conflicts with --audit-image-model")
    return [candidate]
