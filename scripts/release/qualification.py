"""Evidence-backed platform and runtime qualification policy."""

import re
from datetime import date


QUALIFICATION_LIFECYCLE = (
    "install", "discovery", "first_valid_run", "cancellation", "resume",
    "report_generation", "bundle_export", "upgrade", "rollback", "uninstall",
)
LIFECYCLE_STATES = {"passed", "failed", "not_tested"}
SUPPORT_LEVELS = {"supported", "experimental", "unverified"}
MAX_QUALIFICATION_RELEASE_AGE = 1
QUALIFICATION_MATRIX: tuple[dict, ...] = ()
QUALIFICATION_TARGETS = (
    {"platform": "macos", "architecture": "arm64", "runtime": "llamacpp", "backend": "metal"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm"},
    {"platform": "linux", "architecture": "aarch64", "runtime": "llamacpp", "backend": "cuda"},
    {"platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda"},
    {"platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "vulkan"},
    {"platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda"},
    {"platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm"},
    {"platform": "linux", "architecture": "aarch64", "runtime": "vllm", "backend": "cuda"},
    {"platform": "wsl2", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda"},
)

ENTRY_KEYS = {
    "id", "platform", "architecture", "runtime", "runtime_version", "backend",
    "qualified_at", "suite_version", "lifecycle", "known_failures", "evidence",
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
    for key in ("architecture", "runtime", "runtime_version", "backend"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError(f"qualification {key} must be non-empty text")
    try:
        date.fromisoformat(entry["qualified_at"])
    except (TypeError, ValueError):
        raise ValueError("qualification date must use YYYY-MM-DD") from None
    release_number(entry["suite_version"])
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


def derive_support_level(entry: dict | None, current_version: str) -> str:
    if entry is None:
        return "unverified"
    validate_qualification_entry(entry)
    states = set(entry["lifecycle"].values())
    if states == {"passed"} and not qualification_is_stale(entry, current_version):
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
            entry["runtime_version"], entry["backend"],
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
                        entries=QUALIFICATION_MATRIX) -> dict | None:
    matches = [entry for entry in entries if (
        entry["platform"], entry["architecture"], entry["runtime"], entry["backend"]
    ) == (platform, architecture, runtime, backend)]
    if runtime_version is not None:
        matches = [entry for entry in matches if entry["runtime_version"] == runtime_version]
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
                           current_version: str, entries=QUALIFICATION_MATRIX) -> dict:
    platform = platform_name(system, wsl=wsl)
    architecture = normalize_architecture(architecture)
    evidence = qualification_entry(
        platform, architecture, runtime, backend, runtime_version, entries,
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
