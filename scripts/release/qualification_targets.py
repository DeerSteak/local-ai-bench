"""Explicit platform targets for ordinary benchmark qualification runs."""


TARGETS = (
    {"id": "macos-m5-pro-llamacpp-metal", "platform": "macos", "architecture": "arm64", "runtime": "llamacpp", "backend": "metal", "accelerator": "M5 Pro"},
    {"id": "geforce-windows-llamacpp-cuda", "platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"id": "radeon-windows-llamacpp-vulkan", "platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "vulkan", "accelerator": "AMD Radeon"},
    {"id": "intel-arc-windows-llamacpp-vulkan", "platform": "windows", "architecture": "x86_64", "runtime": "llamacpp", "backend": "vulkan", "accelerator": "Intel Arc Pro B65"},
    {"id": "geforce-wsl2-llamacpp-cuda", "platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"id": "geforce-wsl2-vllm-cuda", "platform": "wsl2", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA GeForce"},
    {"id": "radeon-wsl2-llamacpp-rocm", "platform": "wsl2", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
    {"id": "radeon-wsl2-vllm-rocm", "platform": "wsl2", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm", "accelerator": "Radeon RX 9060 XT"},
    {"id": "nvidia-linux-llamacpp-cuda", "platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA"},
    {"id": "nvidia-linux-vllm-cuda", "platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA"},
    {"id": "ryzen-ai-halo-llamacpp-rocm", "platform": "linux", "architecture": "x86_64", "runtime": "llamacpp", "backend": "rocm", "accelerator": "Radeon 8060S"},
    {"id": "ryzen-ai-halo-vllm-rocm", "platform": "linux", "architecture": "x86_64", "runtime": "vllm", "backend": "rocm", "accelerator": "Radeon 8060S"},
    {"id": "dgx-spark-llamacpp-cuda", "platform": "linux", "architecture": "aarch64", "runtime": "llamacpp", "backend": "cuda", "accelerator": "NVIDIA GB10"},
    {"id": "dgx-spark-vllm-cuda", "platform": "linux", "architecture": "aarch64", "runtime": "vllm", "backend": "cuda", "accelerator": "NVIDIA GB10"},
)
TARGET_ENGINES = {target["id"]: target["runtime"] for target in TARGETS}


def target_engine(target_id: str) -> str:
    try:
        return TARGET_ENGINES[target_id]
    except KeyError:
        raise ValueError(f"unknown qualification target: {target_id}") from None
