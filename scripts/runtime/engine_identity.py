"""Stable engine identities and their shared adapter/model families."""


LLAMACPP = "llamacpp"
LLAMACPP_VULKAN = "llamacpp-vulkan"
VLLM = "vllm"
LLAMACPP_ENGINES = frozenset({LLAMACPP, LLAMACPP_VULKAN})


def engine_family(engine_name: str) -> str:
    return LLAMACPP if engine_name in LLAMACPP_ENGINES else engine_name
