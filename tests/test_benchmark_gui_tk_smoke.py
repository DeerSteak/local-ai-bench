from types import SimpleNamespace

import pytest

from scripts.app.engine_management import EngineManagementSnapshot
from scripts.runtime import config


def test_benchmark_gui_builds_all_tabs_and_controller_wiring(monkeypatch):
    tk = pytest.importorskip("tkinter")
    ttk = pytest.importorskip("tkinter.ttk")
    benchmark_gui = pytest.importorskip("scripts.app.benchmark_gui")
    try:
        probe = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    probe.destroy()

    inventory = {"llm": [], "custom": [], "embedding": [], "image": []}
    engine = type("SmokeEngine", (), {
        "runtime_backend": lambda self, backend, *, cpu_only=False: "cpu" if cpu_only else backend,
    })()
    observed = {}

    monkeypatch.setattr(benchmark_gui, "load_frontend_state", lambda _path: None)
    monkeypatch.setattr(benchmark_gui, "load_setup_config", lambda _path: {})
    monkeypatch.setattr(benchmark_gui, "find_comfyui_installation", lambda **_kwargs: None)
    monkeypatch.setattr(benchmark_gui, "installed_engine_names", lambda: ["llamacpp"])
    monkeypatch.setattr(benchmark_gui, "get_engine", lambda _name: engine)
    monkeypatch.setattr(benchmark_gui, "build_model_inventory", lambda *_args: inventory.copy())
    monkeypatch.setattr(benchmark_gui, "find_llamacpp_tool", lambda _name: None)
    monkeypatch.setattr(
        benchmark_gui, "start_runtime_profile_load",
        lambda engines, hardware_profile, output_queue: output_queue.put(
            benchmark_gui.pending_runtime_profiles(tuple(engines))
        ),
    )
    monkeypatch.setattr(benchmark_gui.Shared, "build_profile", lambda: {
        "hostname": "test host", "os": "Linux 6.17", "arch": "x86_64",
        "ram_gb": 32.0, "backend": "cpu", "timestamp": "now",
    })
    monkeypatch.setattr(benchmark_gui, "available_gpu_split_modes", lambda *_args: ("layer",))
    monkeypatch.setattr(benchmark_gui, "configured_gpu_devices", lambda _setup: [])
    monkeypatch.setattr(
        benchmark_gui, "collect_engine_management",
        lambda *_args: EngineManagementSnapshot([], []),
    )
    monkeypatch.setattr(
        "scripts.app.benchmark_gui_screens.history_actions.discover_results",
        lambda _path: ([], []),
    )
    monkeypatch.setattr(
        benchmark_gui, "refresh_tk_layout",
        lambda _root: observed.__setitem__("initial_layout_refreshed", True),
    )

    class ImmediateThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(
        "scripts.app.engine_management.threading",
        SimpleNamespace(Thread=ImmediateThread),
    )

    def inspect_instead_of_mainloop(root):
        root.update_idletasks()
        notebook = next(child for child in root.winfo_children() if isinstance(child, ttk.Notebook))
        engine_tab = root.nametowidget(notebook.tabs()[3])
        engine_buttons = [
            widget for widget in descendants(engine_tab)
            if isinstance(widget, ttk.Button) and widget.cget("text") in {
                "Refresh", "Copy Diagnostics", "Update / Rebuild llama.cpp",
                "Update vLLM", "Cancel Operation",
            }
        ]
        observed["title"] = root.title()
        observed["tabs"] = [notebook.tab(tab, "text") for tab in notebook.tabs()]
        observed["bindings"] = {
            sequence: bool(root.bind_all(sequence))
            for sequence in (
                "<MouseWheel>", "<Button-4>", "<Button-5>", "<Tab>",
                "<Shift-Tab>", "<ISO_Left_Tab>",
            )
        }
        observed["space_bindings"] = {
            widget_class: bool(root.bind_class(widget_class, "<space>"))
            for widget_class in ("TButton", "TCheckbutton", "TRadiobutton")
        }
        observed["engine_actions"] = [
            widget.cget("text") for widget in sorted(engine_buttons, key=lambda item: item.winfo_rootx())
        ]
        root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", inspect_instead_of_mainloop)

    assert benchmark_gui.run_benchmark_gui() == 0
    assert observed == {
        "title": f"Local AI Bench v{config.VERSION}",
        "tabs": ["Configuration", "Run Log", "Result History", "Engine Management"],
        "bindings": {
            "<MouseWheel>": True, "<Button-4>": True, "<Button-5>": True,
            "<Tab>": True, "<Shift-Tab>": True, "<ISO_Left_Tab>": True,
        },
        "space_bindings": {"TButton": True, "TCheckbutton": True, "TRadiobutton": True},
        "engine_actions": [
            "Refresh", "Copy Diagnostics", "Update / Rebuild llama.cpp",
            "Update vLLM", "Cancel Operation",
        ],
        "initial_layout_refreshed": True,
    }


def descendants(widget):
    children = list(widget.winfo_children())
    return [*children, *(item for child in children for item in descendants(child))]
