import os

import pytest

from scripts.setup import setup_selection
from scripts.setup.setup_selection import (
    additional_disk_space_needed, qualification_model_selection, save_hf_token, select_models,
    selected_cleanup_names, toggle_all_models,
)


def test_qualification_model_selection_is_the_smallest_complete_engine_set():
    llm, images, embeddings = qualification_model_selection("llamacpp")
    assert [model["tag"] for model in llm] == ["gemma3:1b-it-q4_K_M"]
    assert [model["short"] for model in images] == ["sd15"]
    assert [model["tag"] for model in embeddings] == ["nomic-embed-text"]
    vllm_llm, vllm_images, _ = qualification_model_selection("vllm")
    assert [model["tag"] for model in vllm_llm] == ["granite4.1:3b-q4_K_M"]
    assert vllm_llm[0]["vllm_tool_parser"] == "granite4"
    assert vllm_images == []
    with pytest.raises(ValueError, match="unknown qualification engine"):
        qualification_model_selection("invented")


def test_toggle_all_models_selects_models_without_enabling_cleanup():
    entries = [
        {"kind": "llm", "checked": False},
        {"kind": "image", "checked": True},
        {"kind": "cleanup", "checked": False},
    ]

    toggle_all_models(entries)

    assert [entry["checked"] for entry in entries] == [True, True, False]


def test_toggle_all_models_deselects_models_without_disabling_selected_cleanup():
    entries = [
        {"kind": "llm", "checked": True},
        {"kind": "embed", "checked": True},
        {"kind": "cleanup", "checked": True},
    ]

    toggle_all_models(entries)

    assert [entry["checked"] for entry in entries] == [False, False, True]


def test_selected_cleanup_names_requires_explicitly_checked_cleanup_entry():
    entries = [
        {"kind": "cleanup", "checked": False,
         "item": {"directory_names": ["ignored"]}},
        {"kind": "llm", "checked": True, "item": {}},
        {"kind": "cleanup", "checked": True,
         "item": {"directory_names": ["custom-a", "custom-b"]}},
    ]

    assert selected_cleanup_names(entries) == ["custom-a", "custom-b"]


def test_save_hf_token_strips_value_and_uses_private_permissions(tmp_path):
    path = tmp_path / "credentials" / "hf.txt"

    save_hf_token(path, "  hf_example  ")

    assert path.read_text() == "hf_example\n"
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("token", ["", "   ", "first\nsecond", "first\rsecond"])
def test_save_hf_token_rejects_empty_or_multiline_values(tmp_path, token):
    with pytest.raises(ValueError, match="non-empty single line"):
        save_hf_token(tmp_path / "hf.txt", token)


@pytest.mark.parametrize(
    ("free_gb", "download_gb", "expected"),
    [(100.0, 80.0, 0.0), (80.0, 80.0, 0.0), (72.5, 80.0, 7.5)],
)
def test_additional_disk_space_needed(free_gb, download_gb, expected):
    assert additional_disk_space_needed(free_gb, download_gb) == expected


def test_select_models_accepts_default_catalog_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_selection, "find_non_catalog_model_dirs", lambda _path: [])
    monkeypatch.setattr(setup_selection, "find_non_catalog_vllm_repos", lambda _path: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    llms, images, embeddings, cleanup, vllm_cleanup = select_models(
        engines=("llamacpp",), vllm_cache_home=tmp_path, cancel=lambda: None,
    )

    assert llms
    assert images
    assert embeddings
    assert cleanup == []
    assert vllm_cleanup == []


def test_select_models_displays_complete_z_image_pipeline_size(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(setup_selection, "find_non_catalog_model_dirs", lambda _path: [])
    monkeypatch.setattr(setup_selection, "find_non_catalog_vllm_repos", lambda _path: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    select_models(engines=("llamacpp",), vllm_cache_home=tmp_path, cancel=lambda: None)

    z_image_line = next(line for line in capsys.readouterr().out.splitlines()
                        if "Z-Image Turbo" in line)
    assert "~20.9 GB" in z_image_line


def test_select_models_delegates_cancel(monkeypatch, tmp_path):
    cancelled = []
    replies = iter(["q", ""])
    monkeypatch.setattr(setup_selection, "find_non_catalog_model_dirs", lambda _path: [])
    monkeypatch.setattr(setup_selection, "find_non_catalog_vllm_repos", lambda _path: [])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))

    select_models(
        engines=("llamacpp",), vllm_cache_home=tmp_path,
        cancel=lambda: cancelled.append(True),
    )

    assert cancelled == [True]
