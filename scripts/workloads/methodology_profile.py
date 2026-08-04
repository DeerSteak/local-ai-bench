"""Resolved neutral methodology profile and effective runtime settings."""

from scripts.runtime import config


ENGINE_STAGES = {
    "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
    "conc_tool", "conc_chat",
}


def resolve_methodology_profile(*, engine_name: str, tests, cpu_only: bool) -> dict:
    optimizations = []
    selected = set(tests)
    if engine_name == "llamacpp" and selected & ENGINE_STAGES:
        optimizations.extend((
            f"{engine_name}:batch={config.LLAMACPP_NUM_BATCH}",
            f"{engine_name}:kv_cache={config.LLAMACPP_KV_CACHE_TYPE}",
            f"{engine_name}:gpu_layers={'0' if cpu_only else 'auto'}",
            f"{engine_name}:flash_attention=on",
        ))
    if selected & {"llamabench", "llamabenchconc"}:
        optimizations.append(
            f"llama.cpp:native_gpu_layers={'0' if cpu_only else config.LLAMABENCH_FULL_OFFLOAD_NGL}"
        )
    if "img" in selected:
        optimizations.append("comfyui:dynamic_vram=disabled")
    return {
        "profile": "neutral-v1",
        "effective_optimizations": optimizations,
    }
