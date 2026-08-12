"""Run Log screen layout and widget ownership."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RunLogScreen:
    frame: Any
    status: Any
    text: Any
    run_actions: Any
    result_actions: Any
    stop_button: Any
    pause_button: Any

    def current_log(self) -> str:
        return self.text.get("1.0", "end-1c")


def build_run_log_screen(notebook, *, tk, ttk, configuration_frame) -> RunLogScreen:
    frame = ttk.Frame(notebook, padding=18)
    notebook.add(frame, text="Run Log")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=1)
    ttk.Label(frame, text="Benchmark run", style="Title.TLabel").grid(
        row=0, column=0, sticky="w",
    )
    status = tk.StringVar(value="No benchmark is running.")
    ttk.Label(frame, textvariable=status).grid(row=1, column=0, sticky="w", pady=(2, 10))
    text = tk.Text(frame, wrap="word", state="disabled", font=("TkFixedFont", 10))
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)
    text.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")
    actions = ttk.Frame(frame)
    actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
    run_actions = ttk.Frame(actions)
    run_actions.pack(fill="x")
    result_actions = ttk.Frame(actions)
    result_actions.pack(fill="x", pady=(8, 0))
    stop_button = ttk.Button(run_actions, text="Stop Benchmark", state="disabled")
    stop_button.pack(side="right")
    pause_button = ttk.Button(run_actions, text="Pause", state="disabled")
    pause_button.pack(side="right", padx=(0, 8))
    ttk.Button(
        run_actions, text="Back to Configuration",
        command=lambda: notebook.select(configuration_frame),
    ).pack(side="left")
    return RunLogScreen(
        frame, status, text, run_actions, result_actions, stop_button, pause_button,
    )
