"""Read-only Engine Management tab helpers."""

import json
import platform
import threading

from scripts.runtime import config
from scripts.runtime.shared import Shared
from scripts.setup.runtime_status import (
    EngineStatus, build_llamacpp_status, build_vllm_status,
)


OWNERSHIP_LABELS = {
    "app_managed": "App managed",
    "system_managed": "System managed",
    "external_server": "External server",
    "platform_launcher": "Platform launcher",
    "missing": "Not installed",
}


def engine_status_lines(status: EngineStatus) -> list[tuple[str, str]]:
    lines = [
        ("Ownership", OWNERSHIP_LABELS.get(status.ownership, status.ownership)),
        ("Location", status.location or "Not found"),
        ("Version", status.version or "Unknown"),
        ("Backend", status.backend or "Unknown"),
        ("Health", status.health.replace("_", " ").title()),
    ]
    lines.extend(
        (key.replace("_", " ").title(), str(value))
        for key, value in status.components.items() if value is not None
    )
    lines.extend(("Warning", warning) for warning in status.warnings)
    return lines


def engine_diagnostics_text(statuses: list[EngineStatus]) -> str:
    return json.dumps(
        {status.engine: status.as_dict() for status in statuses}, indent=2, sort_keys=True,
    )


def collect_engine_statuses(engine_factory, hardware_backend: str) -> list[EngineStatus]:
    llama = engine_factory("llamacpp")
    vllm = engine_factory("vllm")
    is_wsl = Shared.detect_wsl(platform.system(), platform.release())
    return [
        build_llamacpp_status(
            llama.runtime_location(), config.LLAMACPP_DIR,
            llama.runtime_backend(hardware_backend),
        ),
        build_vllm_status(
            vllm.runtime_location(), config.VLLM_VENV,
            vllm.runtime_backend(hardware_backend),
            launcher=vllm.runtime_launcher(), server_url=vllm.external_server_url(),
            is_wsl=is_wsl,
        ),
    ]


def build_engine_management_tab(*, parent, root, tk, ttk, status_loader) -> None:  # pragma: no cover
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)
    header = ttk.Frame(parent)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    ttk.Label(header, text="Engine Management", style="Title.TLabel").pack(side="left")
    state = {"statuses": [], "loading": False}
    status_text = tk.StringVar(value="Select Refresh to inspect installed runtimes.")
    ttk.Label(parent, textvariable=status_text).grid(row=2, column=0, sticky="w", pady=(8, 0))
    body = ttk.Frame(parent)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    def render(statuses):
        for child in body.winfo_children():
            child.destroy()
        for column, status in enumerate(statuses):
            box = ttk.LabelFrame(body, text=status.engine, padding=12)
            box.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
            box.columnconfigure(1, weight=1)
            for row, (label, value) in enumerate(engine_status_lines(status)):
                ttk.Label(box, text=f"{label}:", font=("TkDefaultFont", 10, "bold")).grid(
                    row=row, column=0, sticky="nw", padx=(0, 8), pady=2,
                )
                ttk.Label(box, text=value, wraplength=380).grid(row=row, column=1, sticky="nw", pady=2)

    def refresh_finished(statuses=None, error=None):
        state["loading"] = False
        refresh_button.configure(state="normal")
        if error is not None:
            status_text.set(f"Inspection failed: {error}")
            return
        state["statuses"] = statuses
        render(statuses)
        copy_button.configure(state="normal")
        status_text.set("Runtime inspection complete.")

    def refresh():
        if state["loading"]:
            return
        state["loading"] = True
        refresh_button.configure(state="disabled")
        status_text.set("Inspecting runtimes…")

        def worker():
            try:
                statuses = status_loader()
                root.after(0, lambda: refresh_finished(statuses=statuses))
            except Exception as exc:
                root.after(0, lambda error=exc: refresh_finished(error=error))
        threading.Thread(target=worker, daemon=True).start()

    def copy_diagnostics():
        root.clipboard_clear()
        root.clipboard_append(engine_diagnostics_text(state["statuses"]))
        status_text.set("Diagnostics copied to the clipboard.")

    refresh_button = ttk.Button(header, text="Refresh", command=refresh)
    refresh_button.pack(side="right")
    copy_button = ttk.Button(header, text="Copy Diagnostics", command=copy_diagnostics, state="disabled")
    copy_button.pack(side="right", padx=(0, 8))
    refresh()
