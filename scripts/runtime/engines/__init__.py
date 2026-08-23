"""Pluggable inference-engine registry — see docs/engines.md."""

from scripts.runtime.engines.base import InferenceEngine
from scripts.runtime.engines.llamacpp import LlamaCppEngine
from scripts.runtime.engines.vllm import VllmEngine

_REGISTRY: dict[str, type[InferenceEngine]] = {
    "llamacpp": LlamaCppEngine,
    "vllm": VllmEngine,
}


def get_engine(name: str) -> InferenceEngine:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Unknown inference engine {name!r} — known engines: "
            f"{', '.join(sorted(_REGISTRY))}"
        ) from None


def engine_display_name(name: str) -> str:
    return "llama.cpp" if name == "llamacpp" else name


def installed_engine_names(factory=None) -> list[str]:
    """Registered engines whose runtime is actually present. Falls back to every name
    when none is installed, so a frontend still opens and can report the real problem."""
    factory = factory or get_engine
    installed = [name for name in engine_names() if factory(name).is_installed()]
    return installed or engine_names()


def engine_names() -> list[str]:
    """Every registered engine name, sorted — the set --engine all runs across."""
    return sorted(_REGISTRY)
