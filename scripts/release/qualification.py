"""Evidence-backed platform and runtime qualification policy."""

import re
from datetime import date
from pathlib import Path

from scripts.release.qualification_coverage import (
    SMALLEST_EMBEDDING_MODEL, SMALLEST_IMAGE_MODEL,
    qualification_workloads,
)
from scripts.release.qualification_targets import TARGETS
from scripts.workloads.models import qualification_llm_model


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
        "evidence": [
            "qualification-evidence/macos-m5-pro-llamacpp-metal-b10488-v6/"
            "smoke-result.json",
        ],
    },
    {
        "id": "geforce-wsl2-llamacpp-cuda",
        "platform": "wsl2",
        "architecture": "x86_64",
        "runtime": "llamacpp",
        "runtime_version": "b10488",
        "backend": "cuda",
        "accelerator": (
            "Intel(R) Core(TM) Ultra 7 270K Plus\n"
            "NVIDIA GeForce RTX 5060 Ti 55 GB"
        ),
        "qualified_at": "2026-08-19",
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
        "evidence": [
            "qualification-evidence/geforce-wsl2-llamacpp-cuda-b10488-v7/"
            "smoke-result.json",
        ],
    },
    {
        "id": "radeon-wsl2-llamacpp-rocm",
        "platform": "wsl2",
        "architecture": "x86_64",
        "runtime": "llamacpp",
        "runtime_version": "0.1.2-dev",
        "backend": "rocm",
        "accelerator": (
            "AMD Ryzen 7 5800XT 8-Core Processor\n"
            "AMD Radeon RX 9060 XT 31 GB"
        ),
        "qualified_at": "2026-08-20",
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
        "evidence": [
            "qualification-evidence/radeon-wsl2-llamacpp-rocm/"
            "results_qualification_radeon-wsl2-llamacpp-rocm.json",
        ],
    },
    {
        "id": "geforce-windows-llamacpp-cuda",
        "platform": "windows",
        "architecture": "x86_64",
        "runtime": "llamacpp",
        "runtime_version": "0.1.2-dev",
        "backend": "cuda",
        "accelerator": "NVIDIA GeForce RTX 5060 Ti / 31.8574 GB VRAM",
        "qualified_at": "2026-08-19",
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
        "evidence": [
            "qualification-evidence/geforce-windows-llamacpp-cuda/"
            "results_qualification_geforce-windows-llamacpp-cuda.json",
        ],
    },
    {
        "id": "radeon-8060s-windows-llamacpp-vulkan",
        "platform": "windows",
        "architecture": "x86_64",
        "runtime": "llamacpp",
        "runtime_version": "0.1.2-dev",
        "backend": "vulkan",
        "accelerator": (
            "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S / 127 GB RAM\n"
            "AMD Radeon(TM) 8060S Graphics"
        ),
        "qualified_at": "2026-08-20",
        "suite_version": "6.0-pre8",
        "coverage": {
            "workloads": [
                "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
                "conc_tool", "conc_chat", "sustained", "llamabench", "llamabenchconc", "img",
            ],
            "models": ["gemma3:1b-it-q4_K_M", "nomic-embed-text", "sd15"],
            "notes": (
                "Smallest-model functional coverage for every compatible workload using "
                "native build 10516 (b95502ba9); not performance qualification."
            ),
        },
        "evidence": [
            "qualification-evidence/radeon-windows-llamacpp-vulkan/"
            "results_qualification_radeon-windows-llamacpp-vulkan.json",
        ],
    },
    {
        "id": "dgx-spark-llamacpp-cuda",
        "platform": "linux",
        "architecture": "aarch64",
        "runtime": "llamacpp",
        "runtime_version": "0.1.2-dev",
        "backend": "cuda",
        "accelerator": "NVIDIA GB10 122 GB",
        "qualified_at": "2026-08-19",
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
        "evidence": [
            "qualification-evidence/dgx-spark-llamacpp-cuda/"
            "results_qualification_dgx-spark-llamacpp-cuda.json",
        ],
    },
    {
        "id": "dgx-spark-vllm-cuda",
        "platform": "linux",
        "architecture": "aarch64",
        "runtime": "vllm",
        "runtime_version": "0.27.1",
        "backend": "cuda",
        "accelerator": "NVIDIA GB10 122 GB",
        "qualified_at": "2026-08-19",
        "suite_version": "6.0-pre8",
        "coverage": {
            "workloads": [
                "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
                "conc_tool", "conc_chat", "sustained", "vllmbench",
            ],
            "models": ["granite4.1:3b-q4_K_M", "nomic-embed-text"],
            "notes": (
                "Smallest complete-model functional coverage for every compatible workload; "
                "not performance qualification."
            ),
        },
        "evidence": [
            "qualification-evidence/dgx-spark-vllm-cuda/"
            "results_qualification_dgx-spark-vllm-cuda.json",
        ],
    },
    {
        "id": "ryzen-ai-halo-llamacpp-rocm",
        "platform": "linux",
        "architecture": "x86_64",
        "runtime": "llamacpp",
        "runtime_version": "0.1.2-dev",
        "backend": "rocm",
        "accelerator": "AMD Ryzen AI MAX+ 395 w/ Radeon 8060S 125 GB",
        "qualified_at": "2026-08-20",
        "suite_version": "6.0-pre8",
        "coverage": {
            "workloads": [
                "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
                "conc_tool", "conc_chat", "sustained", "llamabench", "llamabenchconc",
            ],
            "models": ["gemma3:1b-it-q4_K_M", "nomic-embed-text"],
            "notes": (
                "Smallest-model functional coverage for every llama.cpp workload using "
                "native build 10486 (7acdbb1); ComfyUI image generation remains unverified."
            ),
        },
        "evidence": [
            "qualification-evidence/ryzen-ai-halo-llamacpp-rocm/"
            "results_qualification_ryzen-ai-halo-llamacpp-rocm.json",
        ],
    },
    {
        "id": "geforce-rtx-5090-wsl2-llamacpp-cuda",
        "platform": "wsl2",
        "architecture": "x86_64",
        "runtime": "llamacpp",
        "runtime_version": "0.1.2-dev",
        "backend": "cuda",
        "accelerator": (
            "AMD Ryzen 7 9850X3D 8-Core Processor\n"
            "NVIDIA GeForce RTX 5090 51 GB"
        ),
        "qualified_at": "2026-08-19",
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
        "evidence": [
            "qualification-evidence/geforce-rtx-5090-wsl2-llamacpp-cuda/"
            "results_qualification_geforce-wsl2-llamacpp-cuda.json",
        ],
    },
    {
        "id": "geforce-rtx-5090-wsl2-vllm-cuda",
        "platform": "wsl2",
        "architecture": "x86_64",
        "runtime": "vllm",
        "runtime_version": "0.27.1",
        "backend": "cuda",
        "accelerator": (
            "AMD Ryzen 7 9850X3D 8-Core Processor\n"
            "NVIDIA GeForce RTX 5090 51 GB"
        ),
        "qualified_at": "2026-08-19",
        "suite_version": "6.0-pre8",
        "coverage": {
            "workloads": [
                "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
                "conc_tool", "conc_chat", "sustained", "vllmbench",
            ],
            "models": ["granite4.1:3b-q4_K_M", "nomic-embed-text"],
            "notes": (
                "Smallest complete-model functional coverage for every compatible workload; "
                "not performance qualification."
            ),
        },
        "evidence": [
            "qualification-evidence/geforce-rtx-5090-wsl2-vllm-cuda/"
            "results_qualification_geforce-wsl2-vllm-cuda.json",
        ],
    },
)
QUALIFICATION_TARGETS = tuple(
    {key: value for key, value in target.items() if key != "id"} for target in TARGETS
)

ENTRY_KEYS = {
    "id", "platform", "architecture", "runtime", "runtime_version", "backend",
    "accelerator", "qualified_at", "suite_version", "coverage", "evidence",
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
    evidence = entry["evidence"]
    if (not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(item, str) or not item.strip() for item in evidence)):
        raise ValueError("qualification evidence references are invalid")
    if any(Path(item).suffix.lower() != ".json" for item in evidence):
        raise ValueError("qualification evidence must reference ordinary result JSON")


def derive_support_level(entry: dict | None, current_version: str) -> str:
    if entry is None:
        return "unverified"
    validate_qualification_entry(entry)
    required_models = {
        qualification_llm_model(entry["runtime"])["tag"], SMALLEST_EMBEDDING_MODEL,
    }
    required_workloads = set(qualification_workloads(entry["runtime"])) - {"img"}
    complete_coverage = (
        required_workloads <= set(entry["coverage"]["workloads"])
        and required_models <= set(entry["coverage"]["models"])
    )
    if not complete_coverage:
        return "unverified"
    if not qualification_is_stale(entry, current_version):
        return "supported"
    return "experimental"


def derive_image_support_level(entry: dict | None, current_version: str) -> str:
    if entry is None or entry["runtime"] != "llamacpp":
        return "unverified"
    validate_qualification_entry(entry)
    complete_coverage = (
        "img" in entry["coverage"]["workloads"]
        and SMALLEST_IMAGE_MODEL in entry["coverage"]["models"]
    )
    if not complete_coverage:
        return "unverified"
    return "experimental" if qualification_is_stale(entry, current_version) else "supported"


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
            "image_support_level": derive_image_support_level(evidence, current_version),
            "runtime_version": evidence.get("runtime_version") if evidence else None,
            "qualified_at": evidence.get("qualified_at") if evidence else None,
            "suite_version": evidence.get("suite_version") if evidence else None,
            "stale": qualification_is_stale(evidence, current_version) if evidence else False,
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
        "supported": "Complete smallest-model engine qualification recorded for this runtime.",
        "experimental": "Engine qualification is stale; rerun the current smallest-model workload set.",
        "unverified": "No complete smallest-model engine qualification matches this exact runtime.",
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
                "runtime pass complete smallest-model qualification")
    return None
