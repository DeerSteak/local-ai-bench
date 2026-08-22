from scripts.runtime import config

from scripts.workloads.methodology_profile import resolve_methodology_profile


def test_neutral_profile_records_only_settings_for_selected_runtime_paths():
    profile = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm", "llamabench", "img"], cpu_only=False,
    )
    assert profile["profile"] == "neutral-v2"
    assert profile["sampling_profile"]["profile"] == "deterministic-baseline-v1"
    assert profile["effective_optimizations"] == [
        f"llamacpp:batch={config.LLAMACPP_NUM_BATCH}",
        f"llamacpp:kv_cache={config.LLAMACPP_KV_CACHE_TYPE}",
            "llamacpp:gpu_layers=auto",
            "llamacpp:gpu_split=layer",
            "llamacpp:flash_attention=on",
            "llamacpp:repack=enabled",
            f"llama.cpp:native_kv_cache={config.LLAMACPP_KV_CACHE_TYPE}",
            "llama.cpp:native_gpu_split=layer",
            f"llama.cpp:llama_bench_gpu_layers={config.LLAMABENCH_FULL_OFFLOAD_NGL}",
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


def test_single_gpu_profile_records_no_split_with_gpu_offload(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_GPU_SPLIT_MODE", "single")
    optimizations = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm", "llamabench"], cpu_only=False,
    )["effective_optimizations"]
    assert "llamacpp:gpu_layers=auto" in optimizations
    assert "llamacpp:gpu_split=none" in optimizations
    assert "llama.cpp:native_gpu_split=none" in optimizations


def test_no_repack_profile_records_server_and_supported_native_path(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", True)
    optimizations = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm", "llamabench", "llamabenchconc"], cpu_only=False,
    )["effective_optimizations"]
    assert "llamacpp:repack=disabled" in optimizations
    assert "llama.cpp:native_repack=disabled" in optimizations
    assert "llama.cpp:llama_bench_gpu_layers=999" in optimizations
    assert "llama.cpp:llama_batched_bench_gpu_layers=auto" in optimizations


def test_llamabench_profile_does_not_record_unsupported_repack_setting(monkeypatch):
    monkeypatch.setattr(config, "LLAMACPP_NO_REPACK", True)
    optimizations = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llamabench"], cpu_only=False,
    )["effective_optimizations"]
    assert all("repack" not in value for value in optimizations)


def test_batched_bench_profile_records_auto_fit_without_llama_bench_policy():
    optimizations = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llamabenchconc"], cpu_only=False,
    )["effective_optimizations"]
    assert "llama.cpp:llama_batched_bench_gpu_layers=auto" in optimizations
    assert all("llama_bench_gpu_layers" not in value for value in optimizations)


def test_batched_bench_cpu_profile_records_zero_gpu_layers():
    optimizations = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llamabenchconc"], cpu_only=True,
    )["effective_optimizations"]
    assert "llama.cpp:llama_batched_bench_gpu_layers=0" in optimizations


def test_cpu_profile_records_cpu_offload_without_unselected_paths():
    profile = resolve_methodology_profile(
        engine_name="llamacpp", tests=["conv"], cpu_only=True,
    )
    assert "llamacpp:gpu_layers=0" in profile["effective_optimizations"]
    assert all("bench_gpu_layers" not in value for value in profile["effective_optimizations"])
    assert all("comfyui" not in value for value in profile["effective_optimizations"])


def test_non_llamacpp_engine_does_not_inherit_llamacpp_settings():
    profile = resolve_methodology_profile(
        engine_name="future", tests=["llm"], cpu_only=False,
    )
    assert profile == {"profile": "neutral-v2", "effective_optimizations": []}


def test_embedding_only_profile_does_not_claim_text_sampling_controls():
    profile = resolve_methodology_profile(
        engine_name="vllm", tests=["emb"], cpu_only=False,
    )
    assert "sampling_profile" not in profile


def test_text_sampling_profile_maps_to_the_selected_engine():
    llama = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm"], cpu_only=False,
    )["sampling_profile"]
    vllm = resolve_methodology_profile(
        engine_name="vllm", tests=["llm"], cpu_only=False,
    )["sampling_profile"]
    assert llama["semantic_controls"] == vllm["semantic_controls"]
    assert "repeat_penalty" in llama["engine_controls"]
    assert "repetition_penalty" in vllm["engine_controls"]


def test_vllm_profile_records_one_cache_policy_for_server_and_native_workloads():
    profile = resolve_methodology_profile(
        engine_name="vllm", tests=["llm", "conv", "vllmbench"], cpu_only=False,
        vllm_kv_cache_dtype="fp8",
    )
    assert f"vllm:bench_iters={config.VLLMBENCH_ITERS}" in profile["effective_optimizations"]
    assert "vllm:kv_cache=fp8" in profile["effective_optimizations"]


def test_vllm_profile_records_native_mtp_only_for_server_text_generation():
    enabled = resolve_methodology_profile(
        engine_name="vllm", tests=["llm"], cpu_only=False, mtp_enabled=True,
    )
    native_only = resolve_methodology_profile(
        engine_name="vllm", tests=["vllmbench"], cpu_only=False, mtp_enabled=True,
    )
    assert "vllm:native_mtp=on" in enabled["effective_optimizations"]
    assert "vllm:native_mtp=on" not in native_only["effective_optimizations"]


def test_llamacpp_profile_records_native_mtp_only_for_server_text_generation():
    configurations = {
        "qwen3.5:4b-q4_K_M": {
            "num_speculative_tokens": 3, "predictor": "embedded",
        },
    }
    enabled = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llm"], cpu_only=False, mtp_enabled=True,
        mtp_configurations=configurations,
    )
    native_only = resolve_methodology_profile(
        engine_name="llamacpp", tests=["llamabench"], cpu_only=False, mtp_enabled=True,
    )
    assert "llamacpp:native_mtp=on" in enabled["effective_optimizations"]
    assert enabled["mtp_configurations"] == configurations
    assert "llamacpp:native_mtp=on" not in native_only["effective_optimizations"]
    assert "mtp_configurations" not in native_only


def test_sustained_profile_records_sampling_and_native_mtp_identity():
    configurations = {
        "qwen3.5:4b-q4_K_M": {
            "num_speculative_tokens": 3, "predictor": "embedded",
        },
    }
    profile = resolve_methodology_profile(
        engine_name="llamacpp", tests=["sustained"], cpu_only=False,
        mtp_enabled=True, mtp_configurations=configurations,
    )
    assert "llamacpp:native_mtp=on" in profile["effective_optimizations"]
    assert profile["mtp_configurations"] == configurations
    assert profile["sampling_profile"]["profile"] == "deterministic-baseline-v1"


def test_vllm_profile_records_redacted_platform_launcher_overrides():
    profile = resolve_methodology_profile(
        engine_name="vllm", tests=["conv"], cpu_only=False,
        vllm_kv_cache_dtype="fp8",
        vllm_launcher_args=["--gpu-memory-utilization", "0.85", "--api-key", "<secret>"],
    )
    assert profile["effective_optimizations"][-1] == (
        "vllm:launcher_args=--gpu-memory-utilization 0.85 --api-key <secret>"
    )


def test_native_vllmbench_does_not_claim_platform_launcher_overrides():
    profile = resolve_methodology_profile(
        engine_name="vllm", tests=["vllmbench"], cpu_only=False,
        vllm_launcher_args=["--gpu-memory-utilization", "0.85"],
    )
    assert all("launcher_args" not in value for value in profile["effective_optimizations"])
