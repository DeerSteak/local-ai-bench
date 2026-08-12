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
        self.selected: tuple[str, ...] = ()

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *items):
        self.deleted.extend(items)
        self.rows.clear()

    def insert(self, _parent, _position, *, values, tags):
        item = f"row-{len(self.rows)}"
        self.rows[item] = {"values": values, "tags": tags}
        return item

    def selection(self):
        return self.selected

    def index(self, item):
        return self.selected.index(item)


class MessageRecorder:
    def __init__(self, *, confirm=True):
        self.confirm = confirm
        self.errors = []
        self.infos = []
        self.confirmations = []

    def showerror(self, *args, **kwargs):
        self.errors.append((args, kwargs))

    def showinfo(self, *args, **kwargs):
        self.infos.append((args, kwargs))

    def askyesno(self, *args, **kwargs):
        self.confirmations.append((args, kwargs))
        return self.confirm


def build_history_delete_controller(*, active=False, confirm=True):
    tree = FakeTree()
    tree.selected = ("one", "two")
    screen = SimpleNamespace(tree=tree, message=FakeVariable())
    messages = MessageRecorder(confirm=confirm)
    refreshes = []
    controller = HistoryActions(
        screen, root=object(), tk=object(), ttk=object(), filedialog=object(),
        messagebox=messages, process_active=lambda: active,
        review_outbound_metadata=lambda *_args, **_kwargs: None,
        start_recovery=lambda *_args: None,
    )
    controller.item_paths = {
        "one": Path("one.json"), "two": Path("two.json"),
    }
    controller.refresh = lambda: refreshes.append(True)
    return controller, messages, refreshes


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


def test_history_delete_refuses_while_process_is_active(monkeypatch):
    controller, messages, refreshes = build_history_delete_controller(active=True)
    deleted = []
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.delete_multiple_run_artifacts",
        lambda *_args: deleted.append(True),
    )

    controller.delete_selection()

    assert messages.errors[0][0][:2] == (
        "Benchmark active", "Stop the active process first.",
    )
    assert not messages.confirmations
    assert not deleted
    assert not refreshes


def test_history_delete_handles_results_that_no_longer_exist(monkeypatch):
    controller, messages, refreshes = build_history_delete_controller()
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.existing_run_artifacts",
        lambda *_args: [],
    )

    controller.delete_selection()

    assert refreshes == [True]
    assert messages.infos[0][0][:2] == (
        "Delete run", "The selected run no longer exists.",
    )
    assert not messages.confirmations


def test_history_delete_cancel_preserves_all_artifacts(monkeypatch):
    controller, messages, refreshes = build_history_delete_controller(confirm=False)
    artifacts = {
        "one.json": [Path("one.json"), Path("one.sqlite")],
        "two.json": [Path("two.json"), Path("two.log"), Path("two.plan")],
    }
    deleted = []
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.existing_run_artifacts",
        lambda path, _root: artifacts[path.name],
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.delete_multiple_run_artifacts",
        lambda *_args: deleted.append(True),
    )

    controller.delete_selection()

    prompt = messages.confirmations[0][0][1]
    assert "delete 2 selected run(s)" in prompt
    assert "all 5 artifact(s)" in prompt
    assert "one.json (2 artifact(s))" in prompt
    assert "two.json (3 artifact(s))" in prompt
    assert not deleted
    assert not refreshes


def test_history_delete_partial_failure_reports_retained_result(monkeypatch):
    controller, messages, refreshes = build_history_delete_controller()
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.existing_run_artifacts",
        lambda path, _root: [path, path.with_suffix(".sqlite")],
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.delete_multiple_run_artifacts",
        lambda *_args: (
            [Path("one.sqlite")], {Path("two.json"): "permission denied"},
        ),
    )

    controller.delete_selection()

    assert refreshes == [True]
    title, detail = messages.errors[0][0][:2]
    assert title == "Run deletion incomplete"
    assert "Deleted 1 artifact(s)" in detail
    assert "main result was retained" in detail
    assert "two.json: permission denied" in detail


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
