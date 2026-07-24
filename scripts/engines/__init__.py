"""Pluggable inference-engine registry — see docs/engines.md."""

from engines.base import InferenceEngine
from engines.llamacpp import LlamaCppEngine

_REGISTRY: dict[str, type[InferenceEngine]] = {
    "llamacpp": LlamaCppEngine,
}


def get_engine(name: str) -> InferenceEngine:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(
            f"Unknown inference engine {name!r} — known engines: "
            f"{', '.join(sorted(_REGISTRY))}"
        ) from None


def engine_names() -> list[str]:
    """Every registered engine name, sorted — the set --engine all runs across."""
    return sorted(_REGISTRY)
