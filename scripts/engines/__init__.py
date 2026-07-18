"""engines — pluggable inference-engine registry.

get_engine(name) returns a fresh InferenceEngine instance. Ollama and
llama.cpp are implemented today; an MLX engine registers here later without
any change to the benchmark orchestration that drives them through the
interface.
"""

from engines.base import InferenceEngine
from engines.ollama import OllamaEngine
from engines.llamacpp import LlamaCppEngine

_REGISTRY: dict[str, type[InferenceEngine]] = {
    "ollama":   OllamaEngine,
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
