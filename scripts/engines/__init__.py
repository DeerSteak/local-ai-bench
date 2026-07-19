"""engines — pluggable inference-engine registry.

get_engine(name) returns a fresh InferenceEngine instance. llama.cpp is
implemented today; a second engine (e.g. MLX) registers here later without
any change to the benchmark orchestration that drives them through the
interface.
"""

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
    """Every registered engine name, sorted — the full set --engine all runs
    across. Only "llamacpp" today; a second engine appearing in _REGISTRY
    extends this automatically."""
    return sorted(_REGISTRY)
