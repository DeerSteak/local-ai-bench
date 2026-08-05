import json

from scripts.setup.setup_config import (
    available_gpu_split_modes,
    configured_gpu_devices,
    configured_comfyui_dir,
    configured_llamacpp_tool,
    load_setup_config,
    write_setup_config,
)


def test_setup_config_round_trip_contains_paths_without_secrets(tmp_path):
    path = tmp_path / "local_ai_bench_config.json"
    comfyui = tmp_path / "ComfyUI"
    server = tmp_path / "llama-server"
    write_setup_config(
        path, comfyui_dir=comfyui,
        llamacpp_tools={"llama-server": str(server), "llama-bench": None},
        gpu_devices=[{"name": "RTX", "vram_gb": 16.0, "driver": "1"}],
    )

    data = load_setup_config(path)
    assert configured_comfyui_dir(data) == str(comfyui.resolve())
    assert configured_llamacpp_tool(data, "llama-server") == str(server)
    assert configured_llamacpp_tool(data, "llama-bench") is None
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


def test_tensor_split_requires_two_matching_cuda_or_rocm_devices():
    cuda = {"gpu": {"devices": [
        {"backend": "cuda"}, {"backend": "cuda"},
    ]}}
    rocm = {"gpu": {"devices": [
        {"backend": "rocm"}, {"backend": "rocm"},
    ]}}

    assert available_gpu_split_modes(cuda, "cuda") == ("layer", "tensor")
    assert available_gpu_split_modes(rocm, "rocm") == ("layer", "tensor")
    assert available_gpu_split_modes(rocm, "vulkan") == ("layer",)


def test_tensor_split_rejects_single_or_unrecorded_devices():
    single = {"gpu": {"devices": [{"backend": "rocm"}]}}
    legacy = {"gpu": {"devices": [{"name": "AMD GPU"}, {"name": "AMD GPU"}]}}

    assert available_gpu_split_modes(single, "rocm") == ("layer",)
    assert available_gpu_split_modes(legacy, "rocm") == ("layer",)
    assert available_gpu_split_modes({}, "cuda") == ("layer",)
