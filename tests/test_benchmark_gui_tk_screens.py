from types import SimpleNamespace

import pytest

from scripts.app.benchmark_frontend import MenuEntry, TEST_DEFINITIONS, TG_TOKEN_OPTIONS
from scripts.app.benchmark_gui_screens.configuration import build_configuration_screen
from scripts.app.benchmark_gui_screens.engines import build_engine_screen
from scripts.app.benchmark_gui_screens.history import (
    build_history_screen, extend_history_selection, toggle_focused_history_item,
)
from scripts.app.benchmark_gui_screens.progress import ProgressScreen, progress_entries_for_engine
from scripts.app.benchmark_gui_screens.run_log import build_run_log_screen
from scripts.app.benchmark_gui_screens.run_log_actions import RunLogActions
from scripts.app.benchmark_gui_support import (
    progress_event_engine, progress_model_identity, progress_summary_rows,
    update_progress_metrics,
)


@pytest.fixture
def tk_shell():
    tkinter = pytest.importorskip("tkinter")
    ttk = pytest.importorskip("tkinter.ttk")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    style = ttk.Style(root)
    style.configure("Title.TLabel", font=("TkDefaultFont", 12, "bold"))
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    yield root, notebook, tkinter, ttk
    root.update_idletasks()
    root.destroy()


def test_configuration_screen_constructs_and_rerenders_models(tk_shell):
    root, notebook, tk, ttk = tk_shell
    tests = [
        MenuEntry(name, label, family, "Tests", enabled)
        for name, label, family, enabled in TEST_DEFINITIONS
    ]
    test_vars = {entry.value: tk.BooleanVar(root, value=entry.checked) for entry in tests}
    models = [MenuEntry("model-a", "Model A", "llm", "Small", True)]
    model_vars = {"model-a": tk.BooleanVar(root, value=True)}
    screen = build_configuration_screen(
        notebook, tk=tk, ttk=ttk,
        discovery={
            "system": "Test system", "models": "1 model", "storage": "100 GB free",
            "memory_risk": "Low", "runtime": "Available", "comfyui": "Available",
            "issues": [],
        },
        advanced_var=tk.BooleanVar(root), preset_var=tk.StringVar(root, value="Custom"),
        project_status=tk.StringVar(root, value="No project"), preset_names=["Custom"],
        test_vars=test_vars, test_defaults={entry.value: entry.checked for entry in tests},
        custom_tests=tests, model_vars=model_vars, model_defaults={"model-a": True},
        custom_models=models, cap_var=tk.StringVar(root, value="No cap"),
        tg_vars={value: tk.BooleanVar(root, value=True) for value in TG_TOKEN_OPTIONS},
    )
    root.update_idletasks()

    assert notebook.tab(screen.frame, "text") == "Configuration"
    assert set(screen.model_widgets) == {"model-a"}
    models.append(MenuEntry("model-b", "Model B", "llm", "Small", False))
    model_vars["model-b"] = tk.BooleanVar(root, value=False)
    screen.model_defaults["model-b"] = False
    screen.render_model_rows()
    root.update_idletasks()
    assert set(screen.model_widgets) == {"model-a", "model-b"}


def test_configuration_screen_groups_llamacpp_quantizations(tk_shell):
    root, notebook, tk, ttk = tk_shell
    tests = [
        MenuEntry(name, label, family, "Tests", enabled)
        for name, label, family, enabled in TEST_DEFINITIONS
    ]
    test_vars = {entry.value: tk.BooleanVar(root, value=entry.checked) for entry in tests}
    models = [
        MenuEntry(
            "demo:q4", "Demo — Q4 (~1 GB)", "llm", "Small", True,
            base_model="demo", base_label="Demo", variant="Q4", default_variant=True,
        ),
        MenuEntry(
            "demo:q8", "Demo — Q8 (~2 GB)", "llm", "Small", False,
            base_model="demo", base_label="Demo", variant="Q8",
        ),
    ]
    model_vars = {
        entry.value: tk.BooleanVar(root, value=entry.checked) for entry in models
    }
    screen = build_configuration_screen(
        notebook, tk=tk, ttk=ttk,
        discovery={
            "system": "Test", "models": "2", "storage": "100 GB",
            "memory_risk": "Low", "runtime": "Available", "comfyui": "Available",
            "issues": [],
        },
        advanced_var=tk.BooleanVar(root), preset_var=tk.StringVar(root, value="Custom"),
        project_status=tk.StringVar(root, value="No project"), preset_names=["Custom"],
        test_vars=test_vars, test_defaults={entry.value: entry.checked for entry in tests},
        custom_tests=tests,
        model_vars=model_vars, model_defaults={"demo:q4": True, "demo:q8": False},
        custom_models=models, cap_var=tk.StringVar(root, value="No cap"),
        tg_vars={value: tk.BooleanVar(root, value=True) for value in TG_TOKEN_OPTIONS},
    )

    parent_widgets = screen.variant_parent_widgets
    assert parent_widgets is not None
    parent_widgets["demo"].invoke()
    assert all(variable.get() for variable in model_vars.values())
    parent_widgets["demo"].invoke()
    assert not any(variable.get() for variable in model_vars.values())
    screen.set_variant_children_visible(False)
    assert set(screen.model_widgets) == {"demo:q4"}
    assert screen.variant_parent_widgets == {}


def test_configuration_checkbox_text_is_part_of_focusable_control(tk_shell):
    root, notebook, tk, ttk = tk_shell
    tests = [
        MenuEntry(name, label, family, "Tests", enabled)
        for name, label, family, enabled in TEST_DEFINITIONS
    ]
    models = [MenuEntry(
        "demo:q4", "Demo — Q4 (~1 GB)", "llm", "Small", True,
        base_model="demo", base_label="Demo", variant="Q4", default_variant=True,
    )]
    screen = build_configuration_screen(
        notebook, tk=tk, ttk=ttk,
        discovery={
            "system": "Test", "models": "1", "storage": "100 GB",
            "memory_risk": "Low", "runtime": "Available", "comfyui": "Available",
            "issues": [],
        },
        advanced_var=tk.BooleanVar(root), preset_var=tk.StringVar(root, value="Custom"),
        project_status=tk.StringVar(root, value="No project"), preset_names=["Custom"],
        test_vars={entry.value: tk.BooleanVar(root, value=True) for entry in tests},
        test_defaults={entry.value: True for entry in tests}, custom_tests=tests,
        model_vars={"demo:q4": tk.BooleanVar(root, value=True)},
        model_defaults={"demo:q4": True}, custom_models=models,
        cap_var=tk.StringVar(root, value="No cap"),
        tg_vars={value: tk.BooleanVar(root, value=True) for value in TG_TOKEN_OPTIONS},
    )

    assert screen.test_widgets["llm"].cget("text") == "Single-shot LLM"
    assert screen.model_widgets["demo:q4"].cget("text") == "Q4 (~1 GB)"


def test_run_log_screen_constructs_and_navigates_back(tk_shell):
    root, notebook, tk, ttk = tk_shell
    configuration = ttk.Frame(notebook)
    notebook.add(configuration, text="Configuration")
    screen = build_run_log_screen(
        notebook, tk=tk, ttk=ttk, configuration_frame=configuration,
    )
    notebook.select(screen.frame)
    back_button = next(
        child for child in screen.run_actions.winfo_children()
        if child.cget("text") == "Back to Configuration"
    )
    back_button.invoke()
    root.update_idletasks()

    assert notebook.select() == str(configuration)
    assert screen.current_log() == ""
    assert str(screen.stop_button.cget("state")) == "disabled"


def test_history_screen_constructs_selectable_result_table(tk_shell):
    root, notebook, tk, ttk = tk_shell
    screen = build_history_screen(notebook, tk=tk, ttk=ttk)
    item = screen.tree.insert(
        "", "end", values=("now", "system", "complete", "vllm", "default", 2),
    )
    screen.tree.selection_set(item)
    root.update_idletasks()

    assert notebook.tab(screen.frame, "text") == "Result History"
    assert screen.tree.selection() == (item,)
    assert screen.status_filter.get() == "all"
    assert screen.engine_filter.get() == "all"


def test_history_keyboard_selection_can_extend_and_toggle(tk_shell):
    _root, notebook, tk, ttk = tk_shell
    screen = build_history_screen(notebook, tk=tk, ttk=ttk)
    items = [screen.tree.insert("", "end", values=(index,)) for index in range(3)]
    screen.tree.focus(items[0])
    screen.tree.selection_set(items[0])

    assert extend_history_selection(screen.tree, 1) == "break"
    assert screen.tree.selection() == (items[0], items[1])
    assert toggle_focused_history_item(screen.tree) == "break"
    assert screen.tree.selection() == (items[0],)


def test_history_rows_keep_dark_text_on_light_fills_in_dark_mode(tk_shell):
    root, notebook, _tk, ttk = tk_shell
    screen = build_history_screen(notebook, tk=_tk, ttk=ttk)
    root.update_idletasks()

    assert str(screen.tree.tag_configure("history_even", "foreground")) == "#17202a"
    assert str(screen.tree.tag_configure("history_even", "background")) == "#ffffff"
    assert str(screen.tree.tag_configure("history_odd", "foreground")) == "#17202a"
    assert str(screen.tree.tag_configure("history_odd", "background")) == "#e8f1fa"
    style = ttk.Style(root)
    assert style.lookup("History.Treeview", "foreground", ("!selected",)) == "#17202a"
    assert style.lookup("History.Treeview.Heading", "foreground", ("!disabled",)) == "#17202a"
    assert style.lookup("History.Treeview.Heading", "background", ("!disabled",)) == "#dce8f3"
    assert style.lookup("History.Treeview.Heading", "foreground", ("active",)) == "#17202a"


def test_engine_screen_constructs_with_management_controller(tk_shell, monkeypatch):
    root, notebook, _tk, ttk = tk_shell
    controller = object()
    captured = []
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.engines.build_engine_management_tab",
        lambda **kwargs: captured.append(kwargs) or controller,
    )

    frame, built_controller = build_engine_screen(
        notebook, ttk=ttk, root=root, status_loader=lambda: None,
    )
    root.update_idletasks()

    assert notebook.tab(frame, "text") == "Engine Management"
    assert built_controller is controller
    assert captured[0]["parent"] is frame
    assert captured[0]["root"] is root


def test_progress_screen_constructs_and_updates_real_tk_variables(tk_shell):
    root, _notebook, tk, ttk = tk_shell
    screen = ProgressScreen(
        root, tk, ttk, update_progress_metrics, progress_summary_rows,
        progress_event_engine, progress_model_identity,
    )
    entry = MenuEntry("model-a", "Model A", "llm", "Small", True)
    screen.show(
        ["llm"], [entry], ["llamacpp"], {"model-a": {"llamacpp"}},
        show_vram=True,
    )
    screen.update({
        "kind": "model", "stage": "llm", "model": "model-a",
        "status": "complete", "usable": True,
    })
    screen.set_resources({"CPU": "25%", "VRAM": "2 / 8 GB"}, "about 1m")
    root.update_idletasks()

    assert screen.stage_vars[("llamacpp", "llm")].get() == "○ Queued"
    assert screen.model_vars[("llamacpp", "llm", "model-a")].get() == "✓ Complete"
    assert screen.summary_vars["Finished models"].get() == "1 / 1"
    assert screen.resource_vars["CPU"].get() == "25%"
    assert screen.remaining_var.get() == "Remaining time: about 1m"
    screen.window.destroy()


def test_progress_rows_partition_selected_models_by_engine(tk_shell):
    root, _notebook, tk, ttk = tk_shell
    entries = [
        MenuEntry("llama-only", "llama.cpp model", "custom", "Custom", True),
        MenuEntry("vllm-only", "vLLM model", "custom", "Custom", True),
        MenuEntry("both", "Both engines", "llm", "Small", True),
    ]
    owners = {
        "llama-only": {"llamacpp"}, "vllm-only": {"vllm"},
        "both": {"llamacpp", "vllm"},
    }
    assert [entry.value for entry in progress_entries_for_engine(
        entries, "llamacpp", owners,
    )] == ["llama-only", "both"]
    screen = ProgressScreen(
        root, tk, ttk, update_progress_metrics, progress_summary_rows,
        progress_event_engine, progress_model_identity,
    )
    screen.show(["llm"], entries, ["llamacpp", "vllm"], owners, show_vram=False)
    root.update_idletasks()
    assert set(screen.model_vars) == {
        ("llamacpp", "llm", "llama-only"), ("llamacpp", "llm", "both"),
        ("vllm", "llm", "vllm-only"), ("vllm", "llm", "both"),
    }
    assert screen.metrics["total_models"] == 4
    screen.window.destroy()

    screen.show(
        ["llm"], entries[:1], ["llamacpp", "vllm"], owners, show_vram=False,
    )
    root.update_idletasks()
    assert set(screen.stage_vars) == {("llamacpp", "llm")}
    assert set(screen.model_vars) == {("llamacpp", "llm", "llama-only")}
    screen.window.destroy()


def test_progress_rows_separate_mtp_pass_and_filter_incompatible_models(tk_shell):
    root, _notebook, tk, ttk = tk_shell
    entries = [
        MenuEntry("qwen3.5:4b-q4_K_M", "Qwen", "llm", "Small", True),
        MenuEntry("gemma3:1b-it-q4_K_M", "Gemma", "llm", "Small", True),
    ]
    owners = {entry.value: {"vllm"} for entry in entries}
    screen = ProgressScreen(
        root, tk, ttk, update_progress_metrics, progress_summary_rows,
        progress_event_engine, progress_model_identity,
    )
    screen.show(
        ["llm", "emb", "vllmbench", "img"], entries,
        ["vllm · MTP off", "vllm · MTP on"], owners, show_vram=False,
    )
    root.update_idletasks()
    assert ("vllm · MTP on", "llm", "qwen3.5:4b-q4_K_M") in screen.model_vars
    assert ("vllm · MTP on", "llm", "gemma3:1b-it-q4_K_M") not in screen.model_vars
    assert ("vllm · MTP on", "emb") not in screen.stage_vars
    assert ("vllm · MTP on", "vllmbench") not in screen.stage_vars
    screen.window.destroy()

def descendants(widget):
    children = list(widget.winfo_children())
    return [*children, *(item for child in children for item in descendants(child))]


def build_run_log_actions(root, tk, ttk):
    return RunLogActions(
        SimpleNamespace(current_log=lambda: "", result_actions=ttk.Frame(root)),
        root=root, tk=tk, ttk=ttk, filedialog=SimpleNamespace(),
        messagebox=SimpleNamespace(), option_vars={}, active_project={"value": None},
        active_result_paths=lambda: [],
    )


def test_outbound_metadata_review_returns_private_aliases(tk_shell, monkeypatch):
    root, _notebook, tk, ttk = tk_shell
    controller = build_run_log_actions(root, tk, ttk)
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.run_log_actions.outbound_metadata_preview",
        lambda _result: [("System", "Private Host"), ("Hardware", "Private GPU")],
    )
    observed = {}

    def approve_dialog():
        dialog = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
        widgets = descendants(dialog)
        preview = next(widget for widget in widgets if isinstance(widget, tk.Text))
        entries = [widget for widget in widgets if isinstance(widget, ttk.Entry)]
        observed["preview"] = preview.get("1.0", "end-1c")
        entries[0].insert(0, "Published System")
        entries[1].insert(0, "Published Hardware")
        next(
            widget for widget in widgets
            if isinstance(widget, ttk.Button) and widget.cget("text") == "Approve Export"
        ).invoke()

    root.after(0, approve_dialog)
    aliases = controller.review_outbound_metadata({}, "result bundle")

    assert observed["preview"] == "System: Private Host\nHardware: Private GPU"
    assert aliases == {
        "system_alias": "Published System", "hardware_alias": "Published Hardware",
    }


def test_outbound_metadata_review_hides_alias_controls_when_disallowed(tk_shell):
    root, _notebook, tk, ttk = tk_shell
    controller = build_run_log_actions(root, tk, ttk)
    observed = {}

    def cancel_dialog():
        dialog = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
        widgets = descendants(dialog)
        alias_box = next(widget for widget in widgets if isinstance(widget, ttk.LabelFrame))
        observed["manager"] = alias_box.winfo_manager()
        next(
            widget for widget in widgets
            if isinstance(widget, ttk.Button) and widget.cget("text") == "Cancel"
        ).invoke()

    root.after(0, cancel_dialog)
    decision = controller.review_outbound_metadata(
        {"run": {}}, "diagnostic", allow_aliases=False,
    )

    assert observed["manager"] == ""
    assert decision is None


def test_support_preview_requires_explicit_export_approval(tk_shell):
    root, _notebook, tk, ttk = tk_shell
    controller = build_run_log_actions(root, tk, ttk)
    observed = {}

    def approve_dialog():
        dialog = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
        widgets = descendants(dialog)
        preview = next(widget for widget in widgets if isinstance(widget, tk.Text))
        observed["preview"] = preview.get("1.0", "end-1c")
        next(
            widget for widget in widgets
            if isinstance(widget, ttk.Button) and widget.cget("text") == "Export"
        ).invoke()

    root.after(0, approve_dialog)
    accepted = controller.confirm_support_preview({
        "files": ["support.json", "manifest.json"],
        "fields": ["system.os", "diagnostics.error"],
    })

    assert accepted is True
    assert observed["preview"] == (
        "Files:\n  support.json\n  manifest.json\n\n"
        "Fields:\n  system.os\n  diagnostics.error"
    )
