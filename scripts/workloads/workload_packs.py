"""Immutable workload-pack identities and compatibility validation."""

import hashlib
import json

from scripts.stage_registry import STAGE_ORDER


PACK_SCHEMA_VERSION = 1
BUILTIN_PACKS = {
    "core-v1": {
        "schema_version": PACK_SCHEMA_VERSION,
        "id": "core",
        "version": 1,
        "stages": ["llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool", "img"],
        "application_versions": ["4.1"],
        "origin": "builtin",
    },
    "native-crosscheck-v1": {
        "schema_version": PACK_SCHEMA_VERSION,
        "id": "native-crosscheck",
        "version": 1,
        "stages": ["llamabench", "llamabenchconc"],
        "application_versions": ["4.1"],
        "origin": "builtin",
    },
}


def validate_pack(pack, application_version="4.1"):
    """Validate a pack and return a normalized copy with an immutable digest."""
    required = {"schema_version", "id", "version", "stages", "application_versions", "origin"}
    unknown = set(pack) - required
    missing = required - set(pack)
    if missing or unknown:
        raise ValueError(f"invalid workload-pack fields; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if pack["schema_version"] != PACK_SCHEMA_VERSION:
        raise ValueError("unsupported workload-pack schema")
    if not isinstance(pack["id"], str) or not pack["id"].strip():
        raise ValueError("workload pack requires an id")
    if not isinstance(pack["version"], int) or pack["version"] < 1:
        raise ValueError("workload pack requires a positive integer version")
    stages = pack["stages"]
    if not stages or len(stages) != len(set(stages)):
        raise ValueError("workload pack requires unique stages")
    unsupported = set(stages) - set(STAGE_ORDER)
    if unsupported:
        raise ValueError(f"unsupported workload-pack stages: {sorted(unsupported)}")
    if application_version not in pack["application_versions"]:
        raise ValueError(f"workload pack is incompatible with Local AI Bench {application_version}")
    if pack["origin"] not in {"builtin", "custom"}:
        raise ValueError("workload-pack origin must be builtin or custom")
    normalized = json.loads(json.dumps(pack, sort_keys=True))
    encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
    normalized["digest"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return normalized


def load_custom_pack(path, application_version="4.1"):
    """Load a declarative local pack without importing or executing code."""
    with open(path, encoding="utf-8") as handle:
        pack = json.load(handle)
    if pack.get("origin") != "custom":
        raise ValueError("custom workload pack must declare custom origin")
    return validate_pack(pack, application_version)
