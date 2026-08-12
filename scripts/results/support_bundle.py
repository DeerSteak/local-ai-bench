"""Allowlisted, previewable support diagnostics without benchmark content."""

import json
import re
import zipfile
from hashlib import sha256
from pathlib import Path

from scripts.results.canonical_json import canonical_json_bytes
from scripts.results.result_store import as_dict


SUPPORT_SCHEMA_VERSION = 1
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
PROFILE_FIELDS = ("os", "arch", "python", "ram_gb", "backend", "gpu")
DIAGNOSTIC_FIELDS = (
    "error", "reason", "skip_reason", "skip_detail", "timed_out", "timed_out_at",
    "crashed", "crashed_at", "stopped_at", "memory_at_failure",
)
SECRET_PATTERN = re.compile(r"(?i)(hf_[a-z0-9]{12,}|bearer\s+[a-z0-9._-]+)")
PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/)[^\s,;]+")
PUBLIC_SECTIONS = {
    "run", "llm", "llm_conversation", "embeddings", "images", "mcq", "math",
    "reasoning", "code", "tool", "concurrency_tool", "concurrency_chat",
    "llamabench", "llamabenchconc",
}


def _redact(value):
    if not isinstance(value, str):
        return value
    return PATH_PATTERN.sub("<private-path>", SECRET_PATTERN.sub("<secret>", value))


def _diagnostics(value, path="$", output=None):
    output = [] if output is None else output
    if isinstance(value, dict):
        found = {key: _redact(value[key]) for key in DIAGNOSTIC_FIELDS if key in value}
        if found:
            output.append({"path": path, "details": found})
        for key, child in value.items():
            if key not in DIAGNOSTIC_FIELDS:
                child_path = f"{path}.{key}" if path == "$" and key in PUBLIC_SECTIONS else f"{path}.*"
                _diagnostics(child, child_path, output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _diagnostics(child, f"{path}[{index}]", output)
    return output


def build_support_payload(result: dict) -> dict:
    run = as_dict(result.get("run"))
    profile = as_dict(result.get("profile"))
    stages = as_dict(run.get("stages"))
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "application": {
            "version": result.get("version"), "engine": result.get("engine"),
            "result_schema_version": run.get("schema_version"),
        },
        "system": {field: profile[field] for field in PROFILE_FIELDS if field in profile},
        "run": {
            "status": run.get("status"), "reason": _redact(run.get("reason")),
            "requested_tests": run.get("requested_tests", []),
            "plan_id": run.get("plan_id"), "source": run.get("source"),
            "stages": {
                name: {key: value for key, value in stage.items() if key in {
                    "status", "reason", "selected_models", "models_with_results",
                    "models_skipped", "models_failed", "started_at", "finished_at",
                }} for name, stage in stages.items() if isinstance(stage, dict)
            },
        },
        "diagnostics": _diagnostics(result),
    }


def _field_paths(value, path="$", output=None):
    output = [] if output is None else output
    if isinstance(value, dict):
        for key, child in value.items():
            _field_paths(child, f"{path}.{key}", output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _field_paths(child, f"{path}[{index}]", output)
    else:
        output.append(path)
    return output


def preview_support_bundle(result_path: Path) -> dict:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    payload = build_support_payload(result)
    return {"files": ["support.json", "manifest.json"], "fields": _field_paths(payload)}


def export_support_bundle(result_path: Path, bundle_path: Path) -> dict:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    support = canonical_json_bytes(build_support_payload(result))
    manifest = {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "files": {"support.json": {"sha256": sha256(support).hexdigest(), "size": len(support)}},
    }
    entries = {
        "manifest.json": canonical_json_bytes(manifest),
        "support.json": support,
    }
    Path(bundle_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w") as archive:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return manifest
