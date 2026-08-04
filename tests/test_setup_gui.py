import pytest

from scripts.setup.setup_gui import (
    default_model_selection,
    hf_token_review_label,
    mousewheel_scroll_units,
    validate_gui_plan,
)


def test_default_selection_keeps_embeddings_and_respects_memory_limit():
    selection = default_model_selection(1.0)
    assert any(selection.values())
    assert any(not selected for selected in selection.values())


def test_gui_plan_requires_valid_existing_comfyui_path(tmp_path):
    assert validate_gui_plan({"comfyui_mode": "download"}) == []
    assert validate_gui_plan({
        "comfyui_mode": "existing", "comfyui_path": str(tmp_path / "missing"),
    }) == ["The existing ComfyUI path is not usable."]
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    (comfyui / "main.py").touch()
    assert validate_gui_plan({
        "comfyui_mode": "existing", "comfyui_path": str(comfyui),
    }) == []


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        ({"hf_token": "new", "save_hf_token": True}, "provided and saved"),
        ({"hf_token": "new", "save_hf_token": False}, "provided for this run only"),
        ({"hf_token": "", "use_existing_hf_token": True}, "using existing token from HF_TOKEN or hf.txt"),
        ({"hf_token": "", "use_existing_hf_token": False}, "not provided"),
    ],
)
def test_hf_token_review_label_reports_existing_and_new_credentials(plan, expected):
    assert hf_token_review_label(plan) == expected


@pytest.mark.parametrize(
    ("delta", "button", "platform_name", "expected"),
    [
        (1, 0, "darwin", -1),
        (-1, 0, "darwin", 1),
        (120, 0, "win32", -1),
        (-120, 0, "win32", 1),
        (0, 4, "x11", -1),
        (0, 5, "x11", 1),
        (0, 0, "darwin", 0),
    ],
)
def test_mousewheel_scroll_units(delta, button, platform_name, expected):
    assert mousewheel_scroll_units(
        delta=delta, button=button, platform_name=platform_name,
    ) == expected
