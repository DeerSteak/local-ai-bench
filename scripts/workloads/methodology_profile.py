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
        cache_type = (
            "f16" if not cpu_only and config.LLAMACPP_GPU_SPLIT_MODE == "tensor"
            else config.LLAMACPP_KV_CACHE_TYPE
        )
        optimizations.extend((
            f"{engine_name}:batch={config.LLAMACPP_NUM_BATCH}",
            f"{engine_name}:kv_cache={cache_type}",
            f"{engine_name}:gpu_layers={'0' if cpu_only else ('all' if config.LLAMACPP_GPU_SPLIT_MODE == 'tensor' else 'auto')}",
            f"{engine_name}:gpu_split={'none' if cpu_only else config.LLAMACPP_GPU_SPLIT_MODE}",
            f"{engine_name}:flash_attention=on",
        ))
    if "vllmbench" in selected:
        optimizations.append(f"vllm:bench_iters={config.VLLMBENCH_ITERS}")
    if selected & {"llamabench", "llamabenchconc"}:
        optimizations.extend((
            f"llama.cpp:native_gpu_layers={'0' if cpu_only else config.LLAMABENCH_FULL_OFFLOAD_NGL}",
            f"llama.cpp:native_gpu_split={'none' if cpu_only else config.LLAMACPP_GPU_SPLIT_MODE}",
        ))
    if "img" in selected:
        optimizations.append("comfyui:dynamic_vram=disabled")
    return {
        "profile": "neutral-v1",
        "effective_optimizations": optimizations,
    }
