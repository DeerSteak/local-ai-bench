"""Explicit platform targets for ordinary benchmark qualification runs."""

import os
from pathlib import Path


SMART_APP_CONTROL_POLICY_ID = "{0283ac0f-fff1-49ae-ada1-8a933130cad6}"


TARGETS = (
    {"id": "macos-m5-pro-llamacpp-metal", "platform": "macos", "architecture": "arm64", "runtime": "llamacpp", "backend": "metal", "accelerator": "M5 Pro"},
    {"id": "geforce-windows-llamacpp-cuda", "platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"id": "radeon-windows-llamacpp-vulkan", "platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "vulkan", "accelerator": "AMD Radeon"},
    {"id": "intel-arc-windows-llamacpp-sycl", "platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "xpu", "accelerator": "B65"},
    {"id": "geforce-wsl2-llamacpp-cuda", "platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"id": "geforce-wsl2-vllm-cuda", "platform": "wsl2", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"id": "radeon-wsl2-llamacpp-rocm", "platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
    {"id": "nvidia-linux-llamacpp-cuda", "platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA"},
    {"id": "nvidia-linux-vllm-cuda", "platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA"},
    {"id": "radeon-linux-llamacpp-rocm", "platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
    {"id": "radeon-linux-vllm-rocm", "platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
    {"id": "intel-arc-linux-llamacpp-sycl", "platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "xpu", "accelerator": "8086:e222"},
    {"id": "intel-arc-linux-vllm-xpu", "platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "xpu", "accelerator": "8086:e222"},
    {"id": "ryzen-ai-halo-llamacpp-rocm", "platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon 8060S"},
    {"id": "ryzen-ai-halo-vllm-rocm", "platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm", "accelerator": "Radeon 8060S"},
    {"id": "dgx-spark-llamacpp-cuda", "platform": "linux", "architecture": "aarch64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GB10"},
    {"id": "dgx-spark-vllm-cuda", "platform": "linux", "architecture": "aarch64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA GB10"},
)
TARGET_ENGINES = {target["id"]: target["runtime"] for target in TARGETS}


def normalize_architecture(value: str) -> str:
    architecture = str(value or "").strip().lower()
    return {
        "amd64": "x86_64", "x64": "x86_64", "x86-64": "x86_64",
        "aarch64": "arm64",
    }.get(architecture, architecture)


def qualification_target(target_id: str) -> dict:
    try:
        return next(target for target in TARGETS if target["id"] == target_id)
    except StopIteration:
        raise ValueError(f"unknown qualification target: {target_id}") from None


def target_engine(target_id: str) -> str:
    return qualification_target(target_id)["runtime"]


def qualification_host_error(target: dict, *, windows_dir: Path | None = None) -> str | None:
    if target["id"] != "intel-arc-windows-llamacpp-sycl":
        return None
    root = windows_dir or (Path(value) if (value := os.environ.get("WINDIR")) else None)
    if root is None:
        return None
    policy = (
        root / "System32" / "CodeIntegrity" / "CiPolicies" / "Active"
        / f"{SMART_APP_CONTROL_POLICY_ID}.cip"
    )
    if not policy.is_file():
        return None
    return (
        f"target {target['id']} is not supported while Windows Smart App Control policy "
        f"{SMART_APP_CONTROL_POLICY_ID} is enforced; the official llama.cpp SYCL, Vulkan, "
        "and CPU DLLs were blocked. Use the native Linux Intel Arc targets instead"
    )


def qualification_target_errors(target: dict, profile: dict) -> list[str]:
    os_name = str(profile.get("os", "")).split(maxsplit=1)[0].lower()
    actual_platform = (
        "wsl2" if profile.get("wsl") else
        "macos" if os_name == "darwin" else os_name
    )
    actual_architecture = normalize_architecture(profile.get("arch", ""))
    errors = []
    for field, actual in (
        ("platform", actual_platform), ("architecture", actual_architecture),
    ):
        expected = target[field]
        comparable_expected = (
            normalize_architecture(expected) if field == "architecture" else expected
        )
        if actual != comparable_expected:
            errors.append(f"requires {field} {expected}; detected {actual or 'unknown'}")
    expected_backend = target["backend"]
    actual_backend = str(profile.get("backend", "")).lower()
    if expected_backend != "vulkan" and actual_backend != expected_backend:
        errors.append(
            f"requires backend {expected_backend}; detected {actual_backend or 'unknown'}"
        )
    accelerator = str(profile.get("hostname", ""))
    if target["accelerator"].casefold() not in accelerator.casefold():
        errors.append(
            f"requires accelerator identity containing {target['accelerator']!r}; "
            f"detected {accelerator or 'unknown'!r}"
        )
    return errors
