from scripts.runtime import config

from scripts.workloads.methodology_profile import resolve_methodology_profile


def test_neutral_profile_records_only_settings_for_selected_runtime_paths():
    profile = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm", "llamabench", "img"], cpu_only=False,
    )
    assert profile["profile"] == "neutral-v1"
    assert profile["effective_optimizations"] == [
        f"llamacpp:batch={config.LLAMACPP_NUM_BATCH}",
        f"llamacpp:kv_cache={config.LLAMACPP_KV_CACHE_TYPE}",
            "llamacpp:gpu_layers=auto",
            "llamacpp:gpu_split=layer",
            "llamacpp:flash_attention=on",
            f"llama.cpp:native_kv_cache={config.LLAMACPP_KV_CACHE_TYPE}",
            f"llama.cpp:native_gpu_layers={config.LLAMABENCH_FULL_OFFLOAD_NGL}",
            "llama.cpp:native_gpu_split=layer",
        "comfyui:dynamic_vram=disabled",
    ]


def test_tensor_profile_records_split_cache_and_full_offload(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "tensor")
    optimizations = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm"], cpu_only=False,
    )["effective_optimizations"]
    assert "llamacpp:kv_cache=f16" in optimizations
    assert "llamacpp:gpu_layers=all" in optimizations
    assert "llamacpp:gpu_split=tensor" in optimizations


def test_cpu_profile_records_cpu_offload_without_unselected_paths():
    profile = resolve_methodology_profile(
        engine_name="llamacpp", tests=["conv"], cpu_only=True,
    )
    assert "llamacpp:gpu_layers=0" in profile["effective_optimizations"]
    assert all("native_gpu_layers" not in value for value in profile["effective_optimizations"])
    assert all("comfyui" not in value for value in profile["effective_optimizations"])


def test_non_llamacpp_engine_does_not_inherit_llamacpp_settings():
    profile = resolve_methodology_profile(
        engine_name="future", tests=["llm"], cpu_only=False,
    )
    assert profile == {"profile": "neutral-v1", "effective_optimizations": []}


def test_vllm_profile_records_one_cache_policy_for_server_and_native_workloads():
    profile = resolve_methodology_profile(
        engine_name="vllm", tests=["llm", "conv", "vllmbench"], cpu_only=False,
        vllm_kv_cache_dtype="fp8",
    )
    assert f"vllm:bench_iters={config.VLLMBENCH_ITERS}" in profile["effective_optimizations"]
    assert "vllm:kv_cache=fp8" in profile["effective_optimizations"]
