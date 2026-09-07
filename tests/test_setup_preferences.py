import json

import pytest

from scripts.setup import setup_config
from scripts.setup.setup_preferences import (
    model_keys, restore_engine_selection, restore_model_selection, restored_comfyui_options,
    setup_preferences, write_setup_preferences,
)


def test_preferences_round_trip_before_install_and_survive_runtime_handoff(tmp_path):
    path = tmp_path / "config.json"
    keys = model_keys()
    plan = {
        "llm_tags": [keys["llm_tags"][-1]], "embedding_tags": [],
        "image_shorts": [keys["image_shorts"][0]], "engines": ["llamacpp-vulkan"],
        "comfyui_mode": "existing", "comfyui_path": "/chosen/comfyui",
        "save_token_preference": False, "hf_token": "secret-value",
        "cleanup_names": ["delete-me"], "vllm_cleanup_names": ["shared-cache"],
    }
    setup_config.write_setup_config(path, comfyui_dir=None, llamacpp_tools={"llama-server": "/old"})
    write_setup_preferences(path, plan)
    before = setup_config.load_setup_config(path)
    assert before["llama_cpp"] == {"llama-server": "/old"}
    prefs = setup_preferences(before)
    assert prefs["models"][keys["llm_tags"][-1]] is True
    assert all(not prefs["models"][tag] for tag in keys["embedding_tags"])
    assert prefs["engines"] == {"llamacpp": False, "llamacpp-vulkan": True, "vllm": False}
    assert prefs["save_token_preference"] is False
    assert prefs["comfyui_path"] == "/chosen/comfyui"
    setup_config.write_setup_config(path, comfyui_dir=None, llamacpp_tools={"llama-server": "/new"})
    assert setup_preferences(setup_config.load_setup_config(path)) == prefs
    text = path.read_text()
    assert all(value not in text for value in ("secret-value", "delete-me", "shared-cache", "hf_token"))


@pytest.mark.parametrize("saved", [None, [], "bad", {"version": 99}, {"version": True}])
def test_missing_or_unknown_preferences_are_ignored(saved):
    assert setup_preferences({"setup_preferences": saved}) == {}


def test_malformed_fields_are_ignored_independently():
    result = setup_preferences({"setup_preferences": {
        "version": 1, "models": {"a": True, "b": 1, "c": "false"}, "engines": [],
        "comfyui_mode": [], "comfyui_path": 42, "save_token_preference": "false",
    }})
    assert result == {"version": 1, "models": {"a": True}}


def test_restore_preserves_deselections_and_new_catalog_defaults():
    assert restore_model_selection(
        {"old": True, "variant": False, "new": True},
        {"models": {"old": False, "variant": True, "removed": True}},
    ) == {"old": False, "variant": True, "new": True}


def test_unavailable_engines_cannot_be_restored_and_at_least_one_remains():
    entries = [
        {"name": "llamacpp", "enabled": True, "checked": True},
        {"name": "vllm", "enabled": False, "checked": False},
        {"name": "llamacpp-vulkan", "enabled": True, "checked": False},
    ]
    restore_engine_selection(entries, {"engines": {"llamacpp": False, "vllm": True}})
    assert [entry["checked"] for entry in entries] == [True, False, False]
    restore_engine_selection(entries, {"engines": {
        "llamacpp": False, "vllm": True, "llamacpp-vulkan": True,
    }})
    assert [entry["checked"] for entry in entries] == [False, False, True]


def test_comfyui_restore_checks_paths_and_explicit_override(tmp_path):
    saved = tmp_path / "saved"
    explicit = tmp_path / "explicit"
    for path in (saved, explicit):
        path.mkdir()
        (path / "main.py").touch()
    prefs = {"comfyui_mode": "existing", "comfyui_path": str(saved)}
    assert restored_comfyui_options(prefs, None) == ("existing", str(saved))
    assert restored_comfyui_options(prefs, saved, str(explicit)) == ("existing", str(explicit))
    assert restored_comfyui_options({"comfyui_mode": "download"}, saved) == ("download", "")
    (saved / "main.py").unlink()
    assert restored_comfyui_options(prefs, None) == ("download", "")
    assert restored_comfyui_options(prefs, explicit) == ("detected", str(explicit))


def test_qualification_neither_reads_nor_overwrites_interactive_preferences(tmp_path):
    path = tmp_path / "config.json"
    write_setup_preferences(path, {"engines": ["llamacpp"]})
    original = path.read_bytes()
    assert setup_preferences(json.loads(original), qualification=True) == {}
    write_setup_preferences(path, {"engines": ["vllm"]}, qualification=True)
    assert path.read_bytes() == original


def test_failed_atomic_write_keeps_previous_selection_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    write_setup_preferences(path, {"engines": ["llamacpp"]})
    original = path.read_bytes()

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(setup_config.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_setup_preferences(path, {"engines": ["vllm"]})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
