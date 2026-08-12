from pathlib import Path
from types import SimpleNamespace

from scripts.app.benchmark_gui_screens.configuration_files import ConfigurationFileActions
from scripts.app.benchmark_gui_screens.configuration_state import ConfigurationStateController
from scripts.app.benchmark_gui_screens.history_actions import HistoryActions
from scripts.app.benchmark_gui_screens.history_process import HistoryProcessActions
from scripts.app.benchmark_gui_screens.run_log_actions import RunLogActions


class FakeVariable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeTree:
    def __init__(self):
        self.rows = {}
        self.deleted = []

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *items):
        self.deleted.extend(items)
        self.rows.clear()

    def insert(self, _parent, _position, *, values, tags):
        item = f"row-{len(self.rows)}"
        self.rows[item] = {"values": values, "tags": tags}
        return item


def test_configuration_state_applies_imported_controls():
    controller = ConfigurationStateController.__new__(ConfigurationStateController)
    controller.test_vars = {"llm": FakeVariable(), "img": FakeVariable(True)}
    controller.model_vars = {"small": FakeVariable(), "large": FakeVariable(True)}
    controller.cap_var = FakeVariable()
    controller.tg_vars = {64: FakeVariable(), 128: FakeVariable(True)}
    controller.option_vars = {"runs": FakeVariable(), "offline": FakeVariable()}
    selected_engines = []
    controller.set_selected_engines = selected_engines.append

    controller.apply_control_values({
        "tests": {"llm": True}, "models": {"small": True}, "engine": "llamacpp,vllm",
        "max_prompt_tokens": "8192", "tg_tokens": [128],
        "options": {"runs": "5", "offline": True},
    })

    assert {name: var.get() for name, var in controller.test_vars.items()} == {
        "llm": True, "img": False,
    }
    assert {name: var.get() for name, var in controller.model_vars.items()} == {
        "small": True, "large": False,
    }
    assert selected_engines == [["llamacpp", "vllm"]]
    assert controller.cap_var.get() == "8192"
    assert [value for value, var in controller.tg_vars.items() if var.get()] == [128]
    assert controller.option_vars["runs"].get() == "5"
    assert controller.option_vars["offline"].get() is True


def test_configuration_file_action_translates_portable_preset():
    applied = []
    controller = ConfigurationFileActions.__new__(ConfigurationFileActions)
    controller.apply_state = applied.append

    controller.apply_portable_preset({"configuration": {
        "tests": ["llm"], "models": {"llm": ["model"], "embedding": [], "image": []},
        "max_prompt_tokens": 8192, "tg_tokens": [64], "options": {"runs": 3},
    }})

    assert applied == [{
        "tests": ["llm"], "models": {"llm": ["model"], "embedding": [], "image": []},
        "max_prompt_tokens": 8192, "tg_tokens": [64], "gui_options": {"runs": 3},
    }]


def test_history_controller_filters_and_maps_visible_rows(monkeypatch):
    tree = FakeTree()
    screen = SimpleNamespace(
        query=FakeVariable("dgx"), status_filter=FakeVariable("complete"),
        engine_filter=FakeVariable("vllm"), tree=tree, message=FakeVariable(),
    )
    controller = HistoryActions.__new__(HistoryActions)
    controller.screen = screen
    controller.entries = {"all": [{"path": Path("hidden.json")}], "visible": []}
    controller.item_paths = {"old": Path("old.json")}
    visible = [{
        "started_at": "2026-01-01", "system": "DGX", "status": "complete",
        "engine": "vllm", "methodology_profile": "default", "models_with_results": 2,
        "path": Path("result.json"),
    }]
    calls = []

    def fake_filter(entries, **filters):
        calls.append((entries, filters))
        return visible

    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.filter_results", fake_filter,
    )

    controller.apply_filters()

    assert calls == [(
        controller.entries["all"], {"query": "dgx", "status": "complete", "engine": "vllm"},
    )]
    assert controller.entries["visible"] == visible
    assert controller.item_paths == {"row-0": Path("result.json")}
    assert tree.rows["row-0"]["tags"] == ("history_even",)
    assert screen.message.get() == "Showing 1 of 1 local results."


def test_history_process_resume_builds_recovery_launch(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    plan = SimpleNamespace(
        stage_order=["llm", "conv"], engine_name="llamacpp",
    )
    launches = []
    controller = HistoryProcessActions(
        root=object(), filedialog=object(),
        messagebox=SimpleNamespace(askyesno=lambda *_args, **_kwargs: True),
        process_active=lambda: False, launch=lambda *args: launches.append(args),
        fallback_fork=lambda *_args: None,
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_process.load_run_plan", lambda _path: plan,
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_process.recovery_executor_command",
        lambda path: ["recover", str(path)],
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_process.recovery_progress_entries",
        lambda loaded: [loaded.engine_name],
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_process.format_recovery_inspection",
        lambda _report: "inspection",
    )

    controller.start("resume", result_path, {"can_resume": True})

    assert launches == [(
        ["recover", str(result_path)], "recovery", [result_path.resolve()],
        "Recovery is running. Completed evidence is preserved.",
        ["llm", "conv"], ["llamacpp"], ["llamacpp"], "Recovery could not start",
    )]


def test_run_log_export_writes_current_text(monkeypatch, tmp_path):
    destination = tmp_path / "export.txt"
    messages = []
    controller = RunLogActions.__new__(RunLogActions)
    controller.screen = SimpleNamespace(current_log=lambda: "benchmark output\n")
    controller.root = object()
    controller.active_result_paths = lambda: []
    controller.filedialog = SimpleNamespace(
        asksaveasfilename=lambda **_kwargs: str(destination),
    )
    controller.messagebox = SimpleNamespace(
        showinfo=lambda *args, **kwargs: messages.append((args, kwargs)),
        showerror=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.run_log_actions.result_paths_for_log",
        lambda _log, _paths: [],
    )

    controller.export_log()

    assert destination.read_text(encoding="utf-8") == "benchmark output\n"
    assert messages[0][0][0] == "Log exported"
