import json
from pathlib import Path

import pytest

from scripts.app.tk_utils import schedule_tk_layout_refresh

from scripts.setup.setup_gui import (
    HF_LOGIN_URL,
    LLM_GROUPS,
    build_setup_plan,
    engine_checkbox_label,
    sudo_notice,
    model_row_label,
    default_model_selection,
    hf_token_review_label,
    license_button_label,
    mousewheel_scroll_units,
    refresh_tk_layout,
    run_setup_wizard_process,
    selected_gui_token,
    setup_review_lines,
    should_save_gui_token,
    token_controls_enabled,
    next_page_index,
    validate_gui_plan,
)


def test_default_selection_keeps_embeddings_and_respects_memory_limit():
    selection = default_model_selection(1.0)
    assert any(selection.values())
    assert any(not selected for selected in selection.values())


def test_quantization_variants_are_visible_with_only_default_preselected():
    selection = default_model_selection(128.0, ["llamacpp"])
    gemma_variants = [
        model for _, models in LLM_GROUPS for model in models
        if model.get("base_model") == "gemma3:1b-it"
    ]

    assert [(model["variant"], selection[model["tag"]]) for model in gemma_variants] == [
        ("Q4_K_M", True), ("Q6_K", False), ("Q8_0", False),
    ]
    assert all(model["download_size"] in model_row_label(model, ["llamacpp"], 128.0)
               for model in gemma_variants)


def test_gui_plan_requires_valid_existing_comfyui_path(tmp_path):
    assert validate_gui_plan({"comfyui_mode": "download", "image_shorts": ["flux"]}) == []
    assert validate_gui_plan({
        "comfyui_mode": "existing", "comfyui_path": str(tmp_path / "missing"),
        "image_shorts": ["flux"],
    }) == ["The existing ComfyUI path is not usable."]
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    (comfyui / "main.py").touch()
    assert validate_gui_plan({
        "comfyui_mode": "existing", "comfyui_path": str(comfyui), "image_shorts": ["flux"],
    }) == []


def test_gui_plan_ignores_comfyui_when_no_image_models_are_selected(tmp_path):
    unusable = {"comfyui_mode": "existing", "comfyui_path": str(tmp_path / "missing")}
    assert validate_gui_plan({**unusable, "image_shorts": []}) == []
    assert validate_gui_plan(unusable) == []
    # An unrelated error still surfaces, so the gate is scoped to the ComfyUI check alone.
    assert validate_gui_plan({**unusable, "image_shorts": [], "engines": []}) == [
        "Select at least one inference engine."]


def test_page_navigation_skips_pages_that_do_not_apply():
    enabled = [True, True, True, False, True]
    assert next_page_index(2, 1, enabled) == 4
    assert next_page_index(4, -1, enabled) == 2
    assert next_page_index(2, 1, [True] * 5) == 3


def test_page_navigation_holds_position_when_nothing_remains():
    assert next_page_index(0, -1, [True, True]) == 0
    assert next_page_index(1, 1, [True, True]) == 1
    assert next_page_index(0, 1, [True, False, False]) == 0


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
    assert license_button_label(url) == "Review license…"


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


def test_scheduled_tk_layout_refresh_does_not_flush_synchronously():
    calls = []

    class Widget:
        def update_idletasks(self):
            calls.append("refresh")

        def after_idle(self, callback):
            calls.append("scheduled")

    schedule_tk_layout_refresh(Widget())

    assert calls == ["scheduled"]


def test_engine_checkbox_label_marks_experimental_and_unavailable_engines():
    plain = {"label": "llama.cpp", "enabled": True, "note": "already installed"}
    assert engine_checkbox_label(plain) == "llama.cpp — already installed"

    experimental = {"label": "vLLM", "enabled": True, "experimental": True, "note": "nightly"}
    assert "(experimental)" in engine_checkbox_label(experimental)

    blocked = {"label": "vLLM", "enabled": False, "experimental": True, "note": "no Windows build"}
    assert "(unavailable on this system)" in engine_checkbox_label(blocked)
    assert "(experimental)" not in engine_checkbox_label(blocked)


def test_gui_plan_requires_at_least_one_engine():
    base = {"comfyui_mode": "download"}
    assert validate_gui_plan({**base, "engines": []}) == ["Select at least one inference engine."]
    assert validate_gui_plan({**base, "engines": ["llamacpp"]}) == []
    assert validate_gui_plan(base) == []


def test_build_setup_plan_filters_models_engines_and_cleanup():
    from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS_XSMALL

    llm_tag = LLM_MODELS_XSMALL[0]["tag"]
    embedding_tag = EMBED_MODELS[0]["tag"]
    image_short = IMAGE_MODELS[0]["short"]
    plan = build_setup_plan(
        model_selection={llm_tag: True, embedding_tag: True, image_short: True},
        cleanup_names=["old"], cleanup_selected=False,
        vllm_cleanup_selection={"keep": False, "remove": True},
        existing_hf_token=True, override_token=True, entered_token=" replacement ",
        save_token=True, comfyui_mode="existing", comfyui_path=" /tmp/ComfyUI ",
        engine_entries=[
            {"name": "llamacpp", "enabled": True},
            {"name": "vllm", "enabled": False},
        ],
        engine_selection={"llamacpp": True, "vllm": True},
    )
    assert plan["llm_tags"] == [llm_tag]
    assert plan["embedding_tags"] == [embedding_tag]
    assert plan["image_shorts"] == [image_short]
    assert plan["cleanup_names"] == []
    assert plan["vllm_cleanup_names"] == ["remove"]
    assert plan["hf_token"] == "replacement"
    assert plan["save_hf_token"] is True
    assert plan["use_existing_hf_token"] is False
    assert plan["comfyui_path"] == "/tmp/ComfyUI"
    assert plan["engines"] == ["llamacpp"]


def test_setup_review_lines_include_only_applicable_details():
    plan = {
        "llm_tags": ["llm"], "embedding_tags": [], "image_shorts": ["image"],
        "cleanup_names": [], "vllm_cleanup_names": ["cached"], "hf_token": "",
        "use_existing_hf_token": True, "comfyui_mode": "existing",
        "comfyui_path": "/tmp/ComfyUI", "engines": ["vllm"],
    }
    lines = setup_review_lines(plan, show_engines=True, sudo_package="build-essential")
    assert "LLM models: 1" in lines
    assert "ComfyUI: existing" in lines
    assert "Engines: vllm" in lines
    assert "ComfyUI path: /tmp/ComfyUI" in lines
    assert any("administrator rights" in line for line in lines)
    assert lines[-1] == "Nothing will be downloaded until you click Install."


def test_setup_review_lines_omit_image_engine_and_sudo_details():
    plan = {
        "llm_tags": [], "embedding_tags": [], "image_shorts": [],
        "cleanup_names": [], "vllm_cleanup_names": [], "hf_token": "",
        "use_existing_hf_token": False, "comfyui_mode": "existing",
        "comfyui_path": "/tmp/ComfyUI", "engines": ["vllm"],
    }
    text = "\n".join(setup_review_lines(plan, show_engines=False, sudo_package=None))
    assert "ComfyUI:" not in text
    assert "ComfyUI path:" not in text
    assert "Engines:" not in text


def test_model_row_label_shows_one_size_for_one_engine():
    model = {"label": "Qwen3.5 9B", "download_size": "~6.2 GB",
             "vllm_download_size": "~12.4 GB", "vllm_repo": "org/q"}
    assert model_row_label(model, ["llamacpp"], 100) == "Qwen3.5 9B  ~6.2 GB"
    assert model_row_label(model, ["vllm"], 100) == "Qwen3.5 9B  ~12.4 GB"


def test_model_row_label_names_both_engines_when_both_are_selected():
    model = {"label": "Qwen3.5 9B", "download_size": "~6.2 GB",
             "vllm_download_size": "~12.4 GB", "vllm_repo": "org/q"}
    label = model_row_label(model, ["llamacpp", "vllm"], 100)
    assert label == "Qwen3.5 9B  llama.cpp ~6.2 GB · vLLM ~12.4 GB"


def test_model_row_label_warns_per_engine_that_will_not_fit():
    model = {"label": "Qwen3.5 9B", "download_size": "~6.2 GB",
             "vllm_download_size": "~12.4 GB", "vllm_repo": "org/q"}
    label = model_row_label(model, ["llamacpp", "vllm"], 12.0)
    assert "⚠ vLLM needs ~14.9 GB" in label
    assert "llama.cpp needs" not in label

    only_vllm = model_row_label(model, ["vllm"], 12.0)
    assert "⚠ needs ~14.9 GB, ~12.0 GB available" in only_vllm


def test_model_row_label_falls_back_when_the_engine_has_no_weights():
    model = {"label": "Some Model", "download_size": "~6.2 GB"}
    assert model_row_label(model, ["vllm"], 100) == "Some Model  ~6.2 GB"


def test_defaults_differ_between_engines_at_the_same_ceiling():
    """Selection follows each engine's own download size. The ceiling has to sit
    between a model's two builds, so it moves whenever the catalog is rebalanced."""
    ceiling = 10.0   # qwen3.5-9b: ~6.2 GB of GGUF against ~9.1 GB of AWQ
    llamacpp = default_model_selection(ceiling, ["llamacpp"])
    vllm = default_model_selection(ceiling, ["vllm"])
    assert vllm != llamacpp
    unchecked = lambda sel: sum(1 for value in sel.values() if not value)
    assert unchecked(vllm) > unchecked(llamacpp), "the larger build of a model fits less often"


def test_engine_specific_sizing_still_separates_some_ceiling():
    """Guards the mechanism rather than one ceiling: if per-engine sizes were ever
    ignored, no ceiling would separate the two engines."""
    separating = [
        c / 2 for c in range(2, 200)
        if default_model_selection(c / 2, ["llamacpp"]) != default_model_selection(c / 2, ["vllm"])
    ]
    assert separating, "no ceiling distinguishes the engines — per-engine sizes are unused"


def test_a_model_fitting_only_one_selected_engine_stays_checked():
    both = default_model_selection(12.0, ["llamacpp", "vllm"])
    llamacpp = default_model_selection(12.0, ["llamacpp"])
    assert both == llamacpp, "still worth downloading for the engine it fits"


def test_model_row_label_renders_every_real_catalog_entry():
    """Image checkpoints carry no download_size; the GUI labels them anyway."""
    from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
    for engines in (["llamacpp"], ["vllm"], ["llamacpp", "vllm"]):
        for model in LLM_MODELS + EMBED_MODELS + IMAGE_MODELS:
            label = model_row_label(model, engines, 117.1)
            assert label.startswith(model["label"])


def test_model_row_label_handles_an_entry_without_any_size():
    assert model_row_label({"label": "SDXL"}, ["llamacpp", "vllm"], 100) == "SDXL"


def test_default_model_selection_covers_every_real_catalog_entry():
    from scripts.workloads.models import IMAGE_MODELS, LLM_MODELS
    for engines in (["llamacpp"], ["vllm"], ["llamacpp", "vllm"]):
        selection = default_model_selection(117.1, engines)
        for model in LLM_MODELS:
            assert model["tag"] in selection
        for model in IMAGE_MODELS:
            assert model["short"] in selection


def test_sudo_notice_only_appears_when_a_privileged_install_will_run():
    assert sudo_notice(["llamacpp", "vllm"], "python3.12-dev").startswith("Installing python3.12-dev")
    assert "password" in sudo_notice(["vllm"], "python3.12-dev")
    assert sudo_notice(["llamacpp"], "python3.12-dev") == "", "no vLLM, no sudo"
    assert sudo_notice(["vllm"], None) == "", "headers already present"
    assert sudo_notice(None, "python3.12-dev") == ""
    assert sudo_notice([], None) == ""
