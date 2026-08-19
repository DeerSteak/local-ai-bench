"""Explicit platform targets for ordinary benchmark qualification runs."""


TARGET_ENGINES = {
    "macos-m5-pro-llamacpp-metal": "llamacpp",
    "geforce-windows-llamacpp-cuda": "llamacpp",
    "radeon-windows-llamacpp-vulkan": "llamacpp",
    "intel-arc-windows-llamacpp-vulkan": "llamacpp",
    "geforce-wsl2-llamacpp-cuda": "llamacpp",
    "geforce-wsl2-vllm-cuda": "vllm",
    "radeon-wsl2-llamacpp-rocm": "llamacpp",
    "radeon-wsl2-vllm-rocm": "vllm",
    "nvidia-linux-llamacpp-cuda": "llamacpp",
    "nvidia-linux-vllm-cuda": "vllm",
    "ryzen-ai-halo-llamacpp-rocm": "llamacpp",
    "ryzen-ai-halo-vllm-rocm": "vllm",
    "dgx-spark-llamacpp-cuda": "llamacpp",
    "dgx-spark-vllm-cuda": "vllm",
}


def target_engine(target_id: str) -> str:
    try:
        return TARGET_ENGINES[target_id]
    except KeyError:
        raise ValueError(f"unknown qualification target: {target_id}") from None
