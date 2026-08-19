"""Evidence-backed platform and runtime qualification policy."""

import re
from datetime import date
from pathlib import Path

from scripts.release.qualification_coverage import (
    SMALLEST_EMBEDDING_MODEL, SMALLEST_IMAGE_MODEL, SMALLEST_LLM_MODEL,
    qualification_workloads,
)


QUALIFICATION_LIFECYCLE = (
    "install", "discovery", "first_valid_run", "cancellation", "resume",
    "report_generation", "bundle_export", "upgrade", "rollback", "uninstall",
)
PLATFORM_QUALIFICATION_STEPS = tuple(
    step for step in QUALIFICATION_LIFECYCLE if step != "uninstall"
)
LIFECYCLE_STATES = {"passed", "failed", "not_tested"}
SUPPORT_LEVELS = {"supported", "experimental", "unverified"}
MAX_QUALIFICATION_RELEASE_AGE = 1
QUALIFICATION_MATRIX: tuple[dict, ...] = (
    {
        "id": "macos-m5-pro-llamacpp-metal",
        "platform": "macos",
        "architecture": "arm64",
        "runtime": "llamacpp",
        "runtime_version": "b10488",
        "backend": "metal",
        "accelerator": "MacBook Pro\nM5 Pro 48 GB",
        "qualified_at": "2026-08-18",
        "suite_version": "6.0-pre8",
        "coverage": {
            "workloads": [
                "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
                "conc_tool", "conc_chat", "sustained", "llamabench", "llamabenchconc", "img",
            ],
            "models": ["gemma3:1b-it-q4_K_M", "nomic-embed-text", "sd15"],
            "notes": (
                "Smallest-model functional coverage for every compatible workload; "
                "not performance qualification."
            ),
        },
        "lifecycle": {step: "passed" for step in QUALIFICATION_LIFECYCLE},
        "known_failures": [],
        "evidence": [
            "qualification-evidence/macos-m5-pro-llamacpp-metal-b10488-v6/"
            "qualification-manifest.json",
        ],
    },
)
QUALIFICATION_TARGETS = (
    {"platform": "macos", "architecture": "arm64", "runtime": "llamacpp", "backend": "metal", "accelerator": "M5 Pro"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon 8060S"},
    {"platform": "linux", "architecture": "aarch64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GB10"},
    {"platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "vulkan", "accelerator": "AMD Radeon"},
    {"platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "vulkan", "accelerator": "Intel Arc Pro B65"},
    {"platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm", "accelerator": "Radeon 8060S"},
    {"platform": "linux", "architecture": "aarch64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA GB10"},
    {"platform": "wsl2", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"platform": "wsl2", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
)

ENTRY_KEYS = {
    "id", "platform", "architecture", "runtime", "runtime_version", "backend",
    "accelerator", "qualified_at", "suite_version", "coverage", "lifecycle",
    "known_failures", "evidence",
}
PLATFORMS = {"macos", "linux", "windows", "wsl2"}


def release_number(version: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)(?:[.-].*)?", version)
    if not match:
        raise ValueError(f"invalid suite version: {version}")
    return int(match.group(1)), int(match.group(2))


def release_age(entry_version: str, current_version: str) -> int:
    entry = release_number(entry_version)
    current = release_number(current_version)
    if entry > current:
        raise ValueError("qualification suite version is newer than the current suite")
    if entry[0] != current[0]:
        return MAX_QUALIFICATION_RELEASE_AGE + 1
    return current[1] - entry[1]


def qualification_is_stale(entry: dict, current_version: str) -> bool:
    return release_age(entry["suite_version"], current_version) > MAX_QUALIFICATION_RELEASE_AGE


def validate_qualification_entry(entry: dict) -> None:
    if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
        raise ValueError("qualification entry has missing or unknown fields")
    if not isinstance(entry["id"], str) or not entry["id"].strip():
        raise ValueError("qualification entry requires an id")
    if entry["platform"] not in PLATFORMS:
        raise ValueError(f"unknown qualification platform: {entry['platform']}")
    for key in ("architecture", "runtime", "runtime_version", "backend", "accelerator"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError(f"qualification {key} must be non-empty text")
    try:
        date.fromisoformat(entry["qualified_at"])
    except (TypeError, ValueError):
        raise ValueError("qualification date must use YYYY-MM-DD") from None
    release_number(entry["suite_version"])
    coverage = entry["coverage"]
    if (not isinstance(coverage, dict) or set(coverage) != {"workloads", "models", "notes"}
            or not isinstance(coverage["workloads"], list) or not coverage["workloads"]
            or not isinstance(coverage["models"], list) or not coverage["models"]
            or not isinstance(coverage["notes"], str)):
        raise ValueError("qualification coverage is invalid")
    lifecycle = entry["lifecycle"]
    if not isinstance(lifecycle, dict) or set(lifecycle) != set(QUALIFICATION_LIFECYCLE):
        raise ValueError("qualification lifecycle is incomplete")
    if any(state not in LIFECYCLE_STATES for state in lifecycle.values()):
        raise ValueError("qualification lifecycle has an invalid state")
    failures = entry["known_failures"]
    if not isinstance(failures, list) or any(
            not isinstance(item, dict) or set(item) != {"step", "detail"}
            or item["step"] not in QUALIFICATION_LIFECYCLE
            or not isinstance(item["detail"], str) or not item["detail"].strip()
            for item in failures):
        raise ValueError("qualification known failures are invalid")
    gaps = {step for step, state in lifecycle.items() if state != "passed"}
    documented = {item["step"] for item in failures}
    if gaps != documented:
        raise ValueError("every incomplete lifecycle step requires one known failure")
    evidence = entry["evidence"]
    if (not isinstance(evidence, list)
            or any(not isinstance(item, str) or not item.strip() for item in evidence)
            or any(state == "passed" for state in lifecycle.values()) and not evidence):
        raise ValueError("qualification evidence references are invalid")
    if all(lifecycle[step] == "passed" for step in PLATFORM_QUALIFICATION_STEPS) and not any(
            Path(item).name == "qualification-manifest.json" for item in evidence):
        raise ValueError("complete platform qualification requires a final evidence manifest")


def derive_support_level(entry: dict | None, current_version: str) -> str:
    if entry is None:
        return "unverified"
    validate_qualification_entry(entry)
    states = {entry["lifecycle"][step] for step in PLATFORM_QUALIFICATION_STEPS}
    required_models = {SMALLEST_LLM_MODEL, SMALLEST_EMBEDDING_MODEL}
    if entry["runtime"] == "llamacpp":
        required_models.add(SMALLEST_IMAGE_MODEL)
    complete_coverage = (
        set(entry["coverage"]["workloads"]) == set(qualification_workloads(entry["runtime"]))
        and required_models <= set(entry["coverage"]["models"])
    )
    if not complete_coverage:
        return "unverified"
    cleanup_attempted = entry["lifecycle"]["uninstall"] != "not_tested"
    if states == {"passed"} and cleanup_attempted and not qualification_is_stale(
            entry, current_version):
        return "supported"
    if "passed" in states:
        return "experimental"
    return "unverified"


def validate_qualification_matrix(entries=QUALIFICATION_MATRIX) -> None:
    ids = []
    identities = []
    for entry in entries:
        validate_qualification_entry(entry)
        ids.append(entry["id"])
        identities.append((
            entry["platform"], entry["architecture"], entry["runtime"],
            entry["runtime_version"], entry["backend"], entry["accelerator"],
        ))
    if len(ids) != len(set(ids)):
        raise ValueError("qualification entry ids must be unique")
    if len(identities) != len(set(identities)):
        raise ValueError("qualification runtime identities must be unique")


def platform_name(system: str, *, wsl: bool = False) -> str:
    if wsl:
        if system != "Linux":
            raise ValueError("WSL2 qualification requires a Linux runtime")
        return "wsl2"
    try:
        return {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}[system]
    except KeyError:
        raise ValueError(f"unsupported qualification operating system: {system}") from None


def qualification_entry(platform: str, architecture: str, runtime: str, backend: str,
                        runtime_version: str | None = None,
                        entries=QUALIFICATION_MATRIX,
                        accelerator: str | None = None) -> dict | None:
    matches = [entry for entry in entries if (
        entry["platform"], entry["architecture"], entry["runtime"], entry["backend"]
    ) == (platform, architecture, runtime, backend)]
    if runtime_version is not None:
        matches = [entry for entry in matches if entry["runtime_version"] == runtime_version]
    if accelerator is not None:
        expected = accelerator.casefold()
        matches = [entry for entry in matches if (
            expected in entry["accelerator"].casefold()
            or entry["accelerator"].casefold() in expected
        )]
    return max(matches, key=lambda item: item["qualified_at"], default=None)


def qualification_rows(current_version: str, *, targets=QUALIFICATION_TARGETS,
                       entries=QUALIFICATION_MATRIX) -> list[dict]:
    validate_qualification_matrix(entries)
    rows = []
    for target in targets:
        evidence = qualification_entry(**target, entries=entries)
        rows.append({
            **target,
            "support_level": derive_support_level(evidence, current_version),
            "runtime_version": evidence.get("runtime_version") if evidence else None,
            "qualified_at": evidence.get("qualified_at") if evidence else None,
            "suite_version": evidence.get("suite_version") if evidence else None,
            "stale": qualification_is_stale(evidence, current_version) if evidence else False,
            "known_failures": evidence.get("known_failures", []) if evidence else [],
        })
    return rows


def normalize_architecture(value: str) -> str:
    return {"AMD64": "x86_64", "x86-64": "x86_64", "arm64": "arm64"}.get(value, value)


def engine_support_profile(*, system: str, architecture: str, wsl: bool, runtime: str,
                           runtime_version: str | None, backend: str,
                           current_version: str, entries=QUALIFICATION_MATRIX,
                           accelerator: str | None = None) -> dict:
    platform = platform_name(system, wsl=wsl)
    architecture = normalize_architecture(architecture)
    evidence = qualification_entry(
        platform, architecture, runtime, backend, runtime_version, entries,
        accelerator=accelerator,
    ) if runtime_version else None
    support_level = derive_support_level(evidence, current_version)
    caveat = {
        "supported": "Full lifecycle qualification recorded for this exact runtime.",
        "experimental": "Qualification is partial or stale; review its recorded gaps.",
        "unverified": "No full lifecycle qualification matches this exact runtime.",
    }[support_level]
    return {
        "support_level": support_level, "caveat": caveat,
        "qualification_id": evidence.get("id") if evidence else None,
        "qualified_at": evidence.get("qualified_at") if evidence else None,
        "suite_version": evidence.get("suite_version") if evidence else None,
        "runtime_version": runtime_version, "platform": platform,
        "architecture": architecture, "backend": backend,
        "stale": qualification_is_stale(evidence, current_version) if evidence else False,
    }


def engine_selection_label(runtime: str, support_profile: dict) -> str:
    level = support_profile["support_level"]
    if runtime == "vllm" and level != "supported":
        return f"vllm — Experimental · {level.capitalize()} qualification"
    label = "llama.cpp" if runtime == "llamacpp" else runtime
    return f"{label} — {level.capitalize()}"


def experimental_acknowledgement_required(runtimes, support_profiles: dict[str, dict]) -> bool:
    return any(
        runtime == "vllm" and support_profiles[runtime]["support_level"] != "supported"
        for runtime in runtimes
    )


def experimental_engine_ack_error(runtimes, acknowledged: bool) -> str | None:
    if "vllm" in runtimes and not acknowledged:
        return ("vLLM requires --ack-experimental-engine until its exact platform and "
                "runtime pass full lifecycle qualification")
    return None
