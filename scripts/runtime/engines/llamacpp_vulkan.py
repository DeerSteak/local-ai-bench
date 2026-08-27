"""Vulkan-specific identity over the shared llama.cpp engine adapter."""

from pathlib import Path

from scripts.runtime import config
from scripts.runtime.engines.llamacpp import LlamaCppEngine


class LlamaCppVulkanEngine(LlamaCppEngine):
    name = "llamacpp-vulkan"
    REQUIRED_BACKEND = "vulkan"

    @classmethod
    def _runtime_dir(cls) -> Path:
        return config.LLAMACPP_VULKAN_DIR
