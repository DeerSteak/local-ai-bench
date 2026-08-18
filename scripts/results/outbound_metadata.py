"""Outbound metadata preview and aliasing without modifying source results."""

import copy

from scripts.results.canonical_json import sha256_json
from scripts.results.result_store import as_dict, validate_json_data


PROFILE_FIELDS = ("hostname", "hardware", "gpu", "cpu", "chip", "processor", "os", "arch", "ram_gb", "backend")
HARDWARE_FIELDS = ("hardware", "gpu", "cpu", "chip", "processor")
TELEMETRY_BLOCK_FIELDS = {"windows", "summary", "headroom", "provenance", "case_id"}
TELEMETRY_WINDOW_FIELDS = {"name", "sample_count", "duration_sec", "channels", "samples"}
TELEMETRY_CHANNEL_FIELDS = {"peak_gb", "mean_gb", "final_gb", "valid_samples"}
TELEMETRY_SAMPLE_FIELDS = {
    "timestamp_sec", "host_ram_used_gb", "process_rss_gb",
    "accelerator_memory_used_gb", "accelerator_memory_total_gb",
}
TELEMETRY_HEADROOM_FIELDS = {"absolute_gb", "fraction", "state", "basis_channel"}
TELEMETRY_PROVENANCE_FIELDS = {"interval_sec", "failed_samples", "channels"}
TELEMETRY_SOURCE_FIELDS = {"source", "failed_samples"}
POWER_BLOCK_FIELDS = {
    "status", "reason", "source", "scope", "energy_joules", "mean_watts",
    "peak_watts", "idle_baseline_watts", "windows", "provenance", "case_id",
    "efficiency",
}
POWER_WINDOW_FIELDS = {
    "name", "sample_count", "duration_sec", "mean_watts", "peak_watts",
    "energy_joules", "samples",
}
POWER_SAMPLE_FIELDS = {"timestamp_sec", "watts"}
POWER_PROVENANCE_FIELDS = {"interval_sec", "failed_samples"}
POWER_EFFICIENCY_FIELDS = {"unit", "work_count", "per_joule"}
TEMPERATURE_BLOCK_FIELDS = {"status", "reason", "windows", "provenance", "case_id"}
TEMPERATURE_CHANNEL_FIELDS = {"peak_c", "mean_c", "final_c", "valid_samples"}
TEMPERATURE_SAMPLE_FIELDS = {
    "timestamp_sec", "soc_package_c", "cpu_package_c", "gpu_die_c", "gpu_hotspot_c",
}
SUSTAINED_MODEL_FIELDS = {
    "context_tokens", "server_context_tokens", "target_duration_sec", "actual_duration_sec",
    "window_sec",
    "ambient_temp_c", "pause_invalidated", "request_count", "valid_request_count",
    "requests", "series", "analysis", "memory", "power", "temperature",
    "skipped", "required_context_tokens", "model_max_context_tokens",
    "label", "unexpected_error", "error", "error_type", "crashed", "crashed_at",
}
SUSTAINED_REQUEST_FIELDS = {
    "start_sec", "end_sec", "generated_tokens", "tokens_per_sec", "validation_errors",
}
SUSTAINED_SERIES_FIELDS = {
    "timestamp_sec", "duration_sec", "tokens", "tokens_per_sec", "host_ram_used_gb",
    "process_rss_gb", "accelerator_memory_used_gb", "power_watts", "soc_package_c", "cpu_package_c",
    "gpu_die_c", "gpu_hotspot_c",
}
SUSTAINED_ANALYSIS_FIELDS = {
    "initial_tokens_per_sec", "steady_state_tokens_per_sec", "retention_ratio",
    "throttle_onset_sec", "performance", "cause", "duration_sec", "window_count",
    "related_trial_drift",
    "ordinal_drift",
}


def _allow_fields(value, allowed):
    return {key: value[key] for key in allowed if key in value}


def _sanitize_channel_map(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: _allow_fields(channel, TELEMETRY_CHANNEL_FIELDS)
        for key, channel in value.items() if isinstance(channel, dict)
    }


def _sanitize_telemetry_block(value: dict) -> dict:
    block = _allow_fields(value, TELEMETRY_BLOCK_FIELDS)
    windows = []
    for window in block.get("windows", []):
        if not isinstance(window, dict):
            continue
        clean = _allow_fields(window, TELEMETRY_WINDOW_FIELDS)
        clean["channels"] = _sanitize_channel_map(clean.get("channels"))
        clean["samples"] = [
            _allow_fields(sample, TELEMETRY_SAMPLE_FIELDS)
            for sample in clean.get("samples", []) if isinstance(sample, dict)
        ]
        windows.append(clean)
    block["windows"] = windows
    block["summary"] = _sanitize_channel_map(block.get("summary"))
    if isinstance(block.get("headroom"), dict):
        block["headroom"] = _allow_fields(block["headroom"], TELEMETRY_HEADROOM_FIELDS)
    if isinstance(block.get("provenance"), dict):
        provenance = _allow_fields(block["provenance"], TELEMETRY_PROVENANCE_FIELDS)
        sources = provenance.get("channels")
        provenance["channels"] = {
            key: _allow_fields(channel, TELEMETRY_SOURCE_FIELDS)
            for key, channel in sources.items()
            if isinstance(sources, dict) and isinstance(channel, dict)
        } if isinstance(sources, dict) else {}
        block["provenance"] = provenance
    return block


def _sanitize_power_block(value: dict) -> dict:
    block = _allow_fields(value, POWER_BLOCK_FIELDS)
    windows = []
    for window in block.get("windows", []):
        if not isinstance(window, dict):
            continue
        clean = _allow_fields(window, POWER_WINDOW_FIELDS)
        clean["samples"] = [
            _allow_fields(sample, POWER_SAMPLE_FIELDS)
            for sample in clean.get("samples", []) if isinstance(sample, dict)
        ]
        windows.append(clean)
    block["windows"] = windows
    if isinstance(block.get("provenance"), dict):
        block["provenance"] = _allow_fields(
            block["provenance"], POWER_PROVENANCE_FIELDS,
        )
    if isinstance(block.get("efficiency"), dict):
        block["efficiency"] = _allow_fields(block["efficiency"], POWER_EFFICIENCY_FIELDS)
    return block


def _sanitize_temperature_block(value: dict) -> dict:
    block = _allow_fields(value, TEMPERATURE_BLOCK_FIELDS)
    windows = []
    for window in block.get("windows", []):
        if not isinstance(window, dict):
            continue
        clean = _allow_fields(window, TELEMETRY_WINDOW_FIELDS)
        channels = clean.get("channels")
        clean["channels"] = {
            key: _allow_fields(channel, TEMPERATURE_CHANNEL_FIELDS)
            for key, channel in channels.items()
            if isinstance(channels, dict) and isinstance(channel, dict)
        } if isinstance(channels, dict) else {}
        clean["samples"] = [
            _allow_fields(sample, TEMPERATURE_SAMPLE_FIELDS)
            for sample in clean.get("samples", []) if isinstance(sample, dict)
        ]
        windows.append(clean)
    block["windows"] = windows
    if isinstance(block.get("provenance"), dict):
        provenance = _allow_fields(block["provenance"], TELEMETRY_PROVENANCE_FIELDS)
        channels = provenance.get("channels")
        provenance["channels"] = {
            key: _allow_fields(channel, TELEMETRY_SOURCE_FIELDS)
            for key, channel in channels.items()
            if isinstance(channels, dict) and isinstance(channel, dict)
        } if isinstance(channels, dict) else {}
        block["provenance"] = provenance
    return block


def _sanitize_sustained_section(value: dict) -> dict:
    result = {}
    for model, entry in value.items():
        if not isinstance(entry, dict):
            continue
        clean = _allow_fields(entry, SUSTAINED_MODEL_FIELDS)
        clean["requests"] = [
            _allow_fields(request, SUSTAINED_REQUEST_FIELDS)
            for request in clean.get("requests", []) if isinstance(request, dict)
        ]
        clean["series"] = [
            _allow_fields(window, SUSTAINED_SERIES_FIELDS)
            for window in clean.get("series", []) if isinstance(window, dict)
        ]
        if isinstance(clean.get("analysis"), dict):
            clean["analysis"] = _allow_fields(clean["analysis"], SUSTAINED_ANALYSIS_FIELDS)
        _sanitize_telemetry(clean)
        result[model] = clean
    return result


def _sanitize_telemetry(value: object) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if key == "sustained" and isinstance(child, dict):
                value[key] = _sanitize_sustained_section(child)
            elif (key == "memory" and isinstance(child, dict)
                    and ("windows" in child or "provenance" in child)):
                value[key] = _sanitize_telemetry_block(child)
            elif (key == "power" and isinstance(child, dict)
                    and ("windows" in child or "provenance" in child)):
                value[key] = _sanitize_power_block(child)
            elif (key == "temperature" and isinstance(child, dict)
                    and ("windows" in child or "provenance" in child)):
                value[key] = _sanitize_temperature_block(child)
            else:
                _sanitize_telemetry(child)
    elif isinstance(value, list):
        for child in value:
            _sanitize_telemetry(child)


def outbound_metadata_preview(result: dict) -> tuple[tuple[str, str], ...]:
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    profile = as_dict(result.get("profile"))
    run = as_dict(result.get("run"))
    rows = [(f"profile.{key}", str(profile[key])) for key in PROFILE_FIELDS if key in profile]
    rows.extend((
        ("engine", str(result.get("engine") or run.get("engine") or "Not recorded")),
        ("application_version", str(result.get("version") or "Not recorded")),
        ("plan_id", str(run.get("plan_id") or "Not recorded")),
    ))
    models = as_dict(run.get("models"))
    for family, entries in sorted(models.items()):
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                if isinstance(entry, dict):
                    rows.append((f"models.{family}[{index}]", str(entry.get("tag") or entry.get("short"))))
    return tuple(rows)


def format_outbound_preview(result: dict) -> str:
    return "\n".join(f"{label}: {value}" for label, value in outbound_metadata_preview(result))


def _source_identity(result: dict) -> dict:
    profile = as_dict(result.get("profile"))
    run = as_dict(result.get("run"))
    return {
        "profile": {key: profile.get(key) for key in PROFILE_FIELDS if key in profile},
        "engine": result.get("engine") or run.get("engine"),
        "models": run.get("models"),
    }


def source_identity_digest(result: dict) -> str:
    return sha256_json(_source_identity(result))


def prepare_outbound_result(result: dict, *, system_alias: str | None = None,
                            hardware_alias: str | None = None) -> dict:
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    for label, alias in (("system", system_alias), ("hardware", hardware_alias)):
        if alias is not None and (not isinstance(alias, str) or not alias.strip()):
            raise ValueError(f"{label} alias must be non-empty text")
    outbound = copy.deepcopy(result)
    _sanitize_telemetry(outbound)
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
    run = as_dict(outbound.get("run"))
    identity = as_dict(run.get("export_identity"))
    return identity.get("source_sha256") == source_identity_digest(source)
