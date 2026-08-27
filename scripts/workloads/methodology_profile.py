"""Resolved neutral methodology profile and effective runtime settings."""

from scripts.runtime import config
from scripts.runtime.sampling import baseline_sampling_profile
from scripts.runtime.engine_identity import engine_family


ENGINE_STAGES = {
    "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
    "conc_tool", "conc_chat", "sustained",
}
TEXT_GENERATION_STAGES = ENGINE_STAGES - {"emb"}


def effective_gpu_split_mode(cpu_only: bool) -> str:
    if cpu_only or config.LLAMACPP_GPU_SPLIT_MODE == "single":
        return "none"
    return config.LLAMACPP_GPU_SPLIT_MODE


def resolve_methodology_profile(*, engine_name: str, tests, cpu_only: bool,
                                vllm_kv_cache_dtype: str = "auto",
                                vllm_launcher_args: list[str] | None = None,
                                mtp_enabled: bool = False,
                                mtp_configurations: dict | None = None) -> dict:
    optimizations = []
    selected = set(tests)
    family = engine_family(engine_name)
    if family == "llamacpp" and selected & ENGINE_STAGES:
        cache_type = (
            "f16" if not cpu_only and config.LLAMACPP_GPU_SPLIT_MODE == "tensor"
            else config.LLAMACPP_KV_CACHE_TYPE
        )
        optimizations.extend((
            f"{engine_name}:batch={config.LLAMACPP_NUM_BATCH}",
            f"{engine_name}:kv_cache={cache_type}",
            f"{engine_name}:gpu_layers={'0' if cpu_only else ('all' if config.LLAMACPP_GPU_SPLIT_MODE == 'tensor' else 'auto')}",
            f"{engine_name}:gpu_split={effective_gpu_split_mode(cpu_only)}",
            f"{engine_name}:flash_attention=on",
            f"{engine_name}:repack={'disabled' if config.LLAMACPP_NO_REPACK else 'enabled'}",
        ))
        if mtp_enabled and selected & TEXT_GENERATION_STAGES:
            optimizations.append("llamacpp:native_mtp=on")
    if "vllmbench" in selected:
        optimizations.append(f"vllm:bench_iters={config.VLLMBENCH_ITERS}")
    if family == "vllm" and selected & (ENGINE_STAGES | {"vllmbench"}):
        optimizations.append(f"vllm:kv_cache={vllm_kv_cache_dtype}")
        if mtp_enabled and selected & TEXT_GENERATION_STAGES:
            optimizations.append("vllm:native_mtp=on")
        if vllm_launcher_args and selected & ENGINE_STAGES:
            optimizations.append(f"vllm:launcher_args={' '.join(vllm_launcher_args)}")
    if selected & {"llamabench", "llamabenchconc"}:
        native_cache = (
            "f16" if not cpu_only and config.LLAMACPP_GPU_SPLIT_MODE == "tensor"
            else config.LLAMACPP_KV_CACHE_TYPE
        )
        optimizations.extend((
            f"llama.cpp:native_kv_cache={native_cache}",
            f"llama.cpp:native_gpu_split={effective_gpu_split_mode(cpu_only)}",
        ))
        if "llamabench" in selected:
            optimizations.append(
                f"llama.cpp:llama_bench_gpu_layers={'0' if cpu_only else config.LLAMABENCH_FULL_OFFLOAD_NGL}"
            )
        if "llamabenchconc" in selected:
            optimizations.extend((
                f"llama.cpp:llama_batched_bench_gpu_layers={'0' if cpu_only else config.LLAMABENCH_CONC_GPU_LAYERS}",
                f"llama.cpp:native_repack={'disabled' if config.LLAMACPP_NO_REPACK else 'enabled'}",
            ))
    if "img" in selected:
        optimizations.append("comfyui:dynamic_vram=disabled")
    resolved = {
        "profile": "neutral-v2",
        "effective_optimizations": optimizations,
    }
    if mtp_enabled and selected & TEXT_GENERATION_STAGES:
        resolved["mtp_configurations"] = dict(mtp_configurations or {})
    if family in {"llamacpp", "vllm"} and selected & TEXT_GENERATION_STAGES:
        resolved["sampling_profile"] = baseline_sampling_profile(family)
    return resolved
