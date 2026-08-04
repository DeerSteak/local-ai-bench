import pytest

from scripts.setup.setup_gui import (
    HF_LOGIN_URL,
    default_model_selection,
    hf_token_review_label,
    license_button_label,
    mousewheel_scroll_units,
    selected_gui_token,
    should_save_gui_token,
    token_controls_enabled,
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


def test_license_button_label_explains_the_link_action():
    url = "https://huggingface.co/example/model"
    assert license_button_label(url) == f"Accept license: {url}"


@pytest.mark.parametrize(
    ("existing", "override", "entered", "expected"),
    [
        (True, False, "replacement", ""),
        (True, True, " replacement ", "replacement"),
        (False, False, " new ", "new"),
    ],
)
def test_selected_gui_token_requires_override_for_existing_credentials(
    existing, override, entered, expected,
):
    assert selected_gui_token(existing, override, entered) == expected


def test_existing_gui_token_is_not_rewritten_when_save_control_is_disabled():
    assert should_save_gui_token("", True) is False
    assert should_save_gui_token("replacement", True) is True


@pytest.mark.parametrize(
    ("existing", "override", "expected"),
    [(False, False, True), (False, True, True), (True, False, False), (True, True, True)],
)
def test_token_controls_require_override_only_when_credential_exists(existing, override, expected):
    assert token_controls_enabled(existing, override) is expected


def test_token_help_opens_the_hugging_face_login_page():
    assert HF_LOGIN_URL == "https://huggingface.co/login"


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
