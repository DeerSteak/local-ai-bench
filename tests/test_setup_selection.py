import os

import pytest

from setup_selection import additional_disk_space_needed, save_hf_token, selected_cleanup_names, toggle_all_models


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
