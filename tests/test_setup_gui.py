import json
from pathlib import Path

import pytest

from scripts.setup.setup_gui import (
    HF_LOGIN_URL,
    vllm_offer_label,
    default_model_selection,
    hf_token_review_label,
    license_button_label,
    mousewheel_scroll_units,
    refresh_tk_layout,
    run_setup_wizard_process,
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


def test_setup_wizard_process_returns_plan_and_removes_handoff_files(monkeypatch, tmp_path):
    created = [tmp_path / "request.json", tmp_path / "response.json"]
    handles = iter([(10, str(created[0])), (11, str(created[1]))])
    monkeypatch.setattr("scripts.setup.setup_gui.tempfile.mkstemp", lambda **_: next(handles))
    monkeypatch.setattr("scripts.setup.setup_gui.os.close", lambda _handle: None)

    def fake_run(command):
        response_path = Path(command[command.index("--response") + 1])
        response_path.write_text(json.dumps({"plan": {"llm_tags": ["model"]}}))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.setup.setup_gui.subprocess.run", fake_run)
    plan = run_setup_wizard_process(
        memory_ceiling_gb=32.0, detected_comfyui=None,
        cleanup_names=["old-model"], existing_hf_token=True,
    )
    assert plan == {"llm_tags": ["model"]}
    assert all(not path.exists() for path in created)


def test_setup_wizard_process_cleans_handoff_files_when_child_fails(monkeypatch, tmp_path):
    created = [tmp_path / "request.json", tmp_path / "response.json"]
    handles = iter([(10, str(created[0])), (11, str(created[1]))])
    monkeypatch.setattr("scripts.setup.setup_gui.tempfile.mkstemp", lambda **_: next(handles))
    monkeypatch.setattr("scripts.setup.setup_gui.os.close", lambda _handle: None)
    monkeypatch.setattr(
        "scripts.setup.setup_gui.subprocess.run",
        lambda _command: type("Result", (), {"returncode": 1})(),
    )
    with pytest.raises(RuntimeError, match="wizard stopped unexpectedly"):
        run_setup_wizard_process(
            memory_ceiling_gb=None, detected_comfyui=None,
            cleanup_names=[], existing_hf_token=False,
        )
    assert all(not path.exists() for path in created)


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


def test_refresh_tk_layout_flushes_now_and_after_idle():
    calls = []

    class Widget:
        def update_idletasks(self):
            calls.append("refresh")

        def after_idle(self, callback):
            calls.append("scheduled")
            callback()

    refresh_tk_layout(Widget())

    assert calls == ["refresh", "scheduled", "refresh"]


def test_vllm_offer_label_marks_experimental_platforms():
    assert vllm_offer_label(None) == ""
    supported = vllm_offer_label({"status": "supported", "reason": "x"})
    assert "experimental" not in supported and "vLLM" in supported
    assert "experimental" in vllm_offer_label({"status": "experimental", "reason": "x"})
