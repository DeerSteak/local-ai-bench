import json

from setup_config import (
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
    )

    data = load_setup_config(path)
    assert configured_comfyui_dir(data) == str(comfyui.resolve())
    assert configured_llamacpp_tool(data, "llama-server") == str(server)
    assert configured_llamacpp_tool(data, "llama-bench") is None
    assert "token" not in path.read_text().lower()


def test_load_rejects_malformed_unknown_and_non_object_config(tmp_path):
    path = tmp_path / "config.json"
    for content in ("{", "[]", json.dumps({"schema_version": 999})):
        path.write_text(content)
        assert load_setup_config(path) == {}
