"""Outbound metadata preview and aliasing without modifying source results."""

import copy
import hashlib
import json

from result_store import validate_json_data


PROFILE_FIELDS = ("hostname", "hardware", "gpu", "cpu", "chip", "processor", "os", "arch", "ram_gb", "backend")
HARDWARE_FIELDS = ("hardware", "gpu", "cpu", "chip", "processor")


def outbound_metadata_preview(result: dict) -> tuple[tuple[str, str], ...]:
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    rows = [(f"profile.{key}", str(profile[key])) for key in PROFILE_FIELDS if key in profile]
    rows.extend((
        ("engine", str(result.get("engine") or run.get("engine") or "Not recorded")),
        ("application_version", str(result.get("version") or "Not recorded")),
        ("plan_id", str(run.get("plan_id") or "Not recorded")),
    ))
    models = run.get("models") if isinstance(run.get("models"), dict) else {}
    for family, entries in sorted(models.items()):
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                if isinstance(entry, dict):
                    rows.append((f"models.{family}[{index}]", str(entry.get("tag") or entry.get("short"))))
    return tuple(rows)


def format_outbound_preview(result: dict) -> str:
    return "\n".join(f"{label}: {value}" for label, value in outbound_metadata_preview(result))


def _source_identity(result: dict) -> dict:
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    run = result.get("run") if isinstance(result.get("run"), dict) else {}
    return {
        "profile": {key: profile.get(key) for key in PROFILE_FIELDS if key in profile},
        "engine": result.get("engine") or run.get("engine"),
        "models": run.get("models"),
    }


def source_identity_digest(result: dict) -> str:
    payload = json.dumps(
        _source_identity(result), allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_outbound_result(result: dict, *, system_alias: str | None = None,
                            hardware_alias: str | None = None) -> dict:
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    for label, alias in (("system", system_alias), ("hardware", hardware_alias)):
        if alias is not None and (not isinstance(alias, str) or not alias.strip()):
            raise ValueError(f"{label} alias must be non-empty text")
    outbound = copy.deepcopy(result)
    profile = outbound.setdefault("profile", {})
    aliases = []
    if system_alias is not None:
        profile["hostname"] = system_alias.strip()
        aliases.append("system")
    if hardware_alias is not None:
        replaced = False
        for field in HARDWARE_FIELDS:
            if field in profile:
                profile[field] = hardware_alias.strip()
                replaced = True
        if not replaced:
            profile["hardware"] = hardware_alias.strip()
        aliases.append("hardware")
    run = outbound.setdefault("run", {})
    run["export_identity"] = {
        "source_sha256": source_identity_digest(result),
        "aliases_applied": aliases,
    }
    validate_json_data(outbound)
    return outbound


def verify_source_identity(outbound: dict, source: dict) -> bool:
    run = outbound.get("run") if isinstance(outbound.get("run"), dict) else {}
    identity = run.get("export_identity") if isinstance(run.get("export_identity"), dict) else {}
    return identity.get("source_sha256") == source_identity_digest(source)
