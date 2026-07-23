from setup_selection import selected_cleanup_names, toggle_all_models


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
