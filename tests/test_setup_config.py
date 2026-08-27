import json

from scripts.setup.setup_config import (
    available_gpu_split_modes,
    configured_gpu_devices,
    configured_comfyui_dir,
    configured_llamacpp_tool,
    configured_llamacpp_vulkan_tool,
    configured_vllm,
    configured_vllm_launcher_args,
    configured_vllm_path,
    load_setup_config,
    vllm_setup_config,
    write_setup_config,
)


def test_setup_config_round_trip_contains_paths_without_secrets(tmp_path):
    path = tmp_path / "local_ai_bench_config.json"
    comfyui = tmp_path / "ComfyUI"
    server = tmp_path / "llama-server"
    vulkan_server = tmp_path / "llama.cpp-vulkan" / "llama-server"
    write_setup_config(
        path, comfyui_dir=comfyui,
        llamacpp_tools={"llama-server": str(server), "llama-bench": None},
        llamacpp_vulkan_tools={"llama-server": str(vulkan_server)},
        gpu_devices=[{"name": "RTX", "vram_gb": 16.0, "driver": "1"}],
    )

    data = load_setup_config(path)
    assert configured_comfyui_dir(data) == str(comfyui.resolve())
    assert configured_llamacpp_tool(data, "llama-server") == str(server)
    assert configured_llamacpp_tool(data, "llama-bench") is None
    assert configured_llamacpp_vulkan_tool(data, "llama-server") == str(vulkan_server)
    assert configured_gpu_devices(data) == [
        {"name": "RTX", "vram_gb": 16.0, "driver": "1"},
    ]
    assert "token" not in path.read_text().lower()


def test_load_rejects_malformed_unknown_and_non_object_config(tmp_path):
    path = tmp_path / "config.json"
    for content in ("{", "[]", json.dumps({"schema_version": 999})):
        path.write_text(content)
        assert load_setup_config(path) == {}


def test_load_accepts_schema_one_without_gpu_topology(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "llama_cpp": {}}))
    assert configured_gpu_devices(load_setup_config(path)) == []


def test_older_setup_config_has_no_vulkan_toolset(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 3, "llama_cpp": {}}))
    assert configured_llamacpp_vulkan_tool(load_setup_config(path), "llama-server") is None


def test_tensor_split_requires_two_matching_cuda_or_rocm_devices():
    cuda = {"gpu": {"devices": [
        {"backend": "cuda"}, {"backend": "cuda"},
    ]}}
    rocm = {"gpu": {"devices": [
        {"backend": "rocm"}, {"backend": "rocm"},
    ]}}

    assert available_gpu_split_modes(cuda, "cuda") == ("single", "layer", "tensor")
    assert available_gpu_split_modes(rocm, "rocm") == ("single", "layer", "tensor")
    assert available_gpu_split_modes(rocm, "vulkan") == ("layer",)


def test_tensor_split_rejects_single_or_unrecorded_devices():
    single = {"gpu": {"devices": [{"backend": "rocm"}]}}
    legacy = {"gpu": {"devices": [{"name": "AMD GPU"}, {"name": "AMD GPU"}]}}

    assert available_gpu_split_modes(single, "rocm") == ("layer",)
    assert available_gpu_split_modes(legacy, "rocm") == ("layer",)
    assert available_gpu_split_modes({}, "cuda") == ("layer",)


# ── vLLM runtime handoff ──

def _write(path, **overrides):
    kwargs = {"comfyui_dir": None, "llamacpp_tools": {}, **overrides}
    write_setup_config(path, **kwargs)
    return load_setup_config(path)


def test_vllm_runtime_round_trips(tmp_path):
    path = tmp_path / "config.json"
    data = _write(path, vllm={
        "executable": "/usr/bin/vllm",
        "launcher": "/usr/bin/vllm-launch",
        "server_url": "http://localhost:8001",
        "launcher_extra_args": ["--gpu-memory-utilization", "0.85"],
    })
    assert configured_vllm_path(data, "launcher") == "/usr/bin/vllm-launch"
    assert configured_vllm_path(data, "server_url") == "http://localhost:8001"
    assert configured_vllm_launcher_args(data) == ["--gpu-memory-utilization", "0.85"]


def test_vllm_setup_handoff_is_shared_and_normalizes_cache_path(tmp_path):
    assert vllm_setup_config(
        executable="/env/bin/vllm", launcher=None, server_url=None,
        launcher_extra_args=["--enforce-eager"], hf_home=tmp_path / "cache",
    ) == {
        "executable": "/env/bin/vllm", "launcher": None, "server_url": None,
        "launcher_extra_args": ["--enforce-eager"], "hf_home": str(tmp_path / "cache"),
    }


def test_absent_vllm_records_an_empty_block(tmp_path):
    data = _write(tmp_path / "config.json")
    assert configured_vllm(data) == {}
    assert configured_vllm_path(data, "launcher") is None
    assert configured_vllm_launcher_args(data) == []


def test_vllm_accessors_tolerate_older_files_and_junk(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 2, "llama_cpp": {}}))
    older = load_setup_config(path)
    assert older, "a schema 2 file must still load"
    assert configured_vllm(older) == {}
    assert configured_vllm_launcher_args(older) == []

    junk = {"vllm": {"launcher": 5, "server_url": "", "launcher_extra_args": ["--ok", 7]}}
    assert configured_vllm_path(junk, "launcher") is None
    assert configured_vllm_path(junk, "server_url") is None
    assert configured_vllm_launcher_args(junk) == ["--ok"]
    assert configured_vllm({"vllm": "nope"}) == {}
