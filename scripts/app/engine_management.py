"""Read-only Engine Management tab helpers."""

import json
import platform
import threading
from dataclasses import dataclass
from typing import Callable

from scripts.runtime import config
from scripts.runtime.shared import Shared
from scripts.setup.runtime_status import (
    EngineStatus, build_llamacpp_status, build_vllm_status,
)
from scripts.setup.custom_models import load_custom_models
from scripts.setup.model_compatibility import (
    ModelCompatibility, inspect_llamacpp_model, inspect_vllm_model,
)
from scripts.setup.runtime_status import runtime_python
from scripts.setup.setup_config import configured_gpu_devices
from scripts.setup.vllm_install import PINNED_PYTHON, VllmSupport, is_dgx_spark
from scripts.setup.runtime_update import RuntimeUpdateControl


OWNERSHIP_LABELS = {
    "app_managed": "App managed",
    "system_managed": "System managed",
    "external_server": "External server",
    "platform_launcher": "Platform launcher",
    "missing": "Not installed",
}


@dataclass(frozen=True)
class EngineManagementSnapshot:
    statuses: list[EngineStatus]
    models: list[ModelCompatibility]


@dataclass(frozen=True)
class EngineManagementController:
    busy: Callable[[], bool]
    cancel: Callable[[], None]


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


def engine_diagnostics_text(statuses: list[EngineStatus],
                            models: list[ModelCompatibility] | None = None) -> str:
    models = models or []
    return json.dumps(
        {
            "engines": {status.engine: status.as_dict() for status in statuses},
            "imported_models": [model.__dict__ for model in models],
        }, indent=2, sort_keys=True,
    )


def vllm_update_support(status: EngineStatus, setup: dict,
                        machine: str) -> VllmSupport | None:
    if status.engine != "vllm" or not status.managed:
        return None
    if status.backend == "rocm":
        return VllmSupport(
            "supported", "rocm_wheel", "Update the app-managed ROCm wheel environment.",
            requires_python=PINNED_PYTHON,
        )
    if status.backend == "cuda":
        names = [str(device.get("name", "")) for device in configured_gpu_devices(setup)]
        if is_dgx_spark(machine, names):
            return VllmSupport(
                "experimental", "nightly_cu130", "Update the app-managed CUDA 13 nightly.",
                requires_python=PINNED_PYTHON,
            )
        return VllmSupport(
            "supported", "cuda_wheel", "Update the app-managed CUDA wheel environment.",
        )
    return None


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


def collect_engine_management(engine_factory, hardware_backend: str) -> EngineManagementSnapshot:
    statuses = collect_engine_statuses(engine_factory, hardware_backend)
    engines = {name: engine_factory(name) for name in ("llamacpp", "vllm")}
    models = []
    for record in load_custom_models():
        engine, tag = record.get("engine"), record.get("tag")
        if not isinstance(tag, str):
            continue
        if engine == "llamacpp":
            paths = engines[engine].model_paths(tag)
            if paths:
                models.append(inspect_llamacpp_model(tag, paths[0]))
        elif engine == "vllm":
            snapshot = engines[engine].model_snapshot(tag)
            if snapshot is not None:
                models.append(inspect_vllm_model(
                    tag, snapshot / "config.json",
                    runtime_python(engines[engine].runtime_location()),
                ))
    return EngineManagementSnapshot(statuses, models)


def build_engine_management_tab(*, parent, root, tk, ttk, messagebox, status_loader,
                                vllm_updater=None, llamacpp_updater=None,
                                llamacpp_update_prompt=None,
                                llamacpp_model_probe=None,
                                run_active=lambda: False) -> EngineManagementController:  # pragma: no cover
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(1, weight=1)
    header = ttk.Frame(parent)
    header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
    ttk.Label(header, text="Engine Management", style="Title.TLabel").pack(side="left")
    state = {"snapshot": EngineManagementSnapshot([], []), "loading": False}
    active_control: list[RuntimeUpdateControl | None] = [None]
    probe_results: dict[tuple[str, str], ModelCompatibility] = {}
    status_text = tk.StringVar(value="Select Refresh to inspect installed runtimes.")
    ttk.Label(parent, textvariable=status_text).grid(row=2, column=0, sticky="w", pady=(8, 0))
    body = ttk.Frame(parent)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)

    output_box = ttk.LabelFrame(parent, text="Operation output", padding=8)
    output_box.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    output_box.columnconfigure(0, weight=1)
    output_text = tk.Text(
        output_box, height=10, wrap="word", state="disabled", font=("TkFixedFont", 10),
    )
    output_scroll = ttk.Scrollbar(output_box, orient="vertical", command=output_text.yview)
    output_text.configure(yscrollcommand=output_scroll.set)
    output_text.grid(row=0, column=0, sticky="ew")
    output_scroll.grid(row=0, column=1, sticky="ns")

    def clear_output():
        output_text.configure(state="normal")
        output_text.delete("1.0", "end")
        output_text.configure(state="disabled")

    def append_output(text):
        def insert():
            output_text.configure(state="normal")
            output_text.insert("end", str(text))
            output_text.see("end")
            output_text.configure(state="disabled")
        root.after(0, insert)

    def render(snapshot):
        for child in body.winfo_children():
            child.destroy()
        for column, status in enumerate(snapshot.statuses):
            box = ttk.LabelFrame(body, text=status.engine, padding=12)
            box.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
            box.columnconfigure(1, weight=1)
            for row, (label, value) in enumerate(engine_status_lines(status)):
                ttk.Label(box, text=f"{label}:", font=("TkDefaultFont", 10, "bold")).grid(
                    row=row, column=0, sticky="nw", padx=(0, 8), pady=2,
                )
                ttk.Label(box, text=value, wraplength=380).grid(row=row, column=1, sticky="nw", pady=2)
        if snapshot.models:
            models_box = ttk.LabelFrame(body, text="Imported model compatibility", padding=12)
            models_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
            for row, model in enumerate(snapshot.models):
                displayed = probe_results.get((model.engine, model.tag)) or model
                text = (f"{displayed.tag} [{displayed.engine}] — "
                        f"{displayed.architecture or 'Unknown architecture'} — "
                        f"{displayed.status.replace('_', ' ')}\n{displayed.detail}")
                ttk.Label(models_box, text=text, wraplength=820).grid(row=row, column=0, sticky="w", pady=3)
                if model.engine == "llamacpp" and llamacpp_model_probe is not None:
                    ttk.Button(
                        models_box, text="Verify",
                        command=lambda selected=model: start_model_probe(selected),
                    ).grid(row=row, column=1, sticky="e", padx=(12, 0))

    def refresh_finished(snapshot=None, error=None):
        state["loading"] = False
        refresh_button.configure(state="normal")
        if error is not None:
            status_text.set(f"Inspection failed: {error}")
            return
        assert snapshot is not None
        state["snapshot"] = snapshot
        render(snapshot)
        copy_button.configure(state="normal")
        status_text.set("Runtime inspection complete.")

    def update_finished(engine, result=None, error=None):
        state["loading"] = False
        active_control[0] = None
        refresh_button.configure(state="normal")
        cancel_button.configure(state="disabled")
        if error is not None:
            status_text.set(f"{engine} update failed: {error}")
            return
        assert result is not None
        if result.success:
            identity = f" ({result.version})" if result.version else ""
            append_output(f"\n{result.detail}{identity}\nRescanning installed runtimes…\n")
            status_text.set(f"{engine} update complete; rescanning installed runtimes…")
            messagebox.showinfo(f"{engine} update", result.detail, parent=root)
            refresh()
        else:
            messagebox.showerror(f"{engine} update", result.detail, parent=root)
            status_text.set(result.detail)

    def probe_finished(model, result=None, error=None):
        state["loading"] = False
        active_control[0] = None
        refresh_button.configure(state="normal")
        cancel_button.configure(state="disabled")
        if error is not None:
            status_text.set(f"Model verification failed: {error}")
            return
        assert result is not None
        probe_results[(model.engine, model.tag)] = result
        render(state["snapshot"])
        status_text.set(result.detail)

    def refresh():
        if state["loading"]:
            return
        state["loading"] = True
        refresh_button.configure(state="disabled")
        status_text.set("Inspecting runtimes…")

        def worker():
            try:
                snapshot = status_loader()
                root.after(0, lambda: refresh_finished(snapshot=snapshot))
            except Exception as exc:
                root.after(0, lambda error=exc: refresh_finished(error=error))
        threading.Thread(target=worker, daemon=True).start()

    def copy_diagnostics():
        root.clipboard_clear()
        snapshot = state["snapshot"]
        root.clipboard_append(engine_diagnostics_text(snapshot.statuses, snapshot.models))
        status_text.set("Diagnostics copied to the clipboard.")

    def update_engine(engine_key, label, updater, prompt, allow_system=False):
        if state["loading"] or updater is None:
            return
        if run_active():
            messagebox.showerror("Benchmark active", "Stop the active benchmark first.", parent=root)
            return
        status = next((item for item in state["snapshot"].statuses if item.engine == engine_key), None)
        if status is None or (not status.managed and not allow_system):
            messagebox.showinfo(
                f"{label} update", f"Only app-managed {label} can be updated here.", parent=root,
            )
            return
        if not messagebox.askyesno(
                f"Update {label}", prompt, parent=root):
            return
        state["loading"] = True
        clear_output()
        append_output(f"Starting {label} update…\n")
        control = RuntimeUpdateControl(append_output)
        active_control[0] = control
        refresh_button.configure(state="disabled")
        cancel_button.configure(state="normal")
        status_text.set(f"Downloading and validating the {label} update…")

        def worker():
            try:
                result = updater(control)
                root.after(0, lambda: update_finished(label, result=result))
            except Exception as exc:
                root.after(0, lambda error=exc: update_finished(label, error=error))
        threading.Thread(target=worker, daemon=True).start()

    def cancel_update():
        control = active_control[0]
        if control is None:
            return
        cancel_button.configure(state="disabled")
        status_text.set("Cancelling update and restoring the prior runtime…")
        control.cancel()

    def start_model_probe(model):
        if state["loading"] or llamacpp_model_probe is None:
            return
        probe = llamacpp_model_probe
        if run_active():
            messagebox.showerror("Benchmark active", "Stop the active benchmark first.", parent=root)
            return
        if not messagebox.askyesno(
                "Verify llama.cpp model",
                f"Load {model.tag} CPU-only with a small context to verify this runtime?",
                parent=root):
            return
        state["loading"] = True
        clear_output()
        append_output(f"Starting verification for {model.tag}…\n")
        control = RuntimeUpdateControl(append_output)
        active_control[0] = control
        refresh_button.configure(state="disabled")
        cancel_button.configure(state="normal")
        status_text.set(f"Verifying {model.tag} with llama.cpp…")

        def worker():
            try:
                result = probe(model.tag, control)
                root.after(0, lambda: probe_finished(model, result=result))
            except Exception as exc:
                root.after(0, lambda error=exc: probe_finished(model, error=error))
        threading.Thread(target=worker, daemon=True).start()

    refresh_button = ttk.Button(header, text="Refresh", command=refresh)
    refresh_button.pack(side="right")
    copy_button = ttk.Button(header, text="Copy Diagnostics", command=copy_diagnostics, state="disabled")
    copy_button.pack(side="right", padx=(0, 8))
    cancel_button = ttk.Button(header, text="Cancel Operation", command=cancel_update, state="disabled")
    cancel_button.pack(side="right", padx=(0, 8))
    if vllm_updater is not None:
        ttk.Button(
            header, text="Update vLLM",
            command=lambda: update_engine(
                "vllm", "vLLM", vllm_updater,
                "Build and validate a new vLLM environment, then replace the current one?",
            ),
        ).pack(side="right", padx=(0, 8))
    if llamacpp_updater is not None:
        ttk.Button(
            header, text="Update / Rebuild llama.cpp",
            command=lambda: update_engine(
                "llamacpp", "llama.cpp", llamacpp_updater,
                llamacpp_update_prompt or "Update the selected llama.cpp installation?",
                allow_system=platform.system() == "Darwin",
            ),
        ).pack(side="right", padx=(0, 8))
    refresh()
    return EngineManagementController(
        busy=lambda: active_control[0] is not None,
        cancel=cancel_update,
    )
