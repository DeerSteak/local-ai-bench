"""Tkinter wizard that collects a complete setup plan before installation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.runtime import hardware
from scripts.workloads.models import (
    EMBED_MODELS,
    IMAGE_MODELS,
    LLM_MODELS_LARGE,
    LLM_MODELS_MEDIUM,
    LLM_MODELS_SMALL,
    LLM_MODELS_XSMALL,
)
from scripts.workloads.model_variants import (
    collapse_variant_selection, expanded_variant_catalog, variant_selection_state,
    variant_selection_target,
)
from scripts.runtime.comfyui_installation import normalize_comfyui_dir
from scripts.setup.engine_selection import LLAMACPP, VLLM
from scripts.setup.model_inventory import (
    engine_fit_report, engine_fit_warnings, fits_any_engine, format_engine_sizes,
)
from scripts.app.tk_utils import mousewheel_scroll_units, refresh_tk_layout
from scripts.app.benchmark_gui_accessibility import (
    configure_explicit_tab_order, configure_keyboard_accessibility,
)


LLM_GROUPS = (
    ("Extra-small LLMs", expanded_variant_catalog(LLM_MODELS_XSMALL)),
    ("Small LLMs", expanded_variant_catalog(LLM_MODELS_SMALL)),
    ("Medium LLMs", expanded_variant_catalog(LLM_MODELS_MEDIUM)),
    ("Large LLMs", expanded_variant_catalog(LLM_MODELS_LARGE)),
)
HF_LOGIN_URL = "https://huggingface.co/login"


def display_wizard_page(pages, index: int) -> None:
    for page_index, page in enumerate(pages):
        if page_index == index:
            page.grid()
        else:
            page.grid_remove()


def focus_scroll_fraction(*, widget_top: int, widget_bottom: int, view_top: int,
                          view_bottom: int, content_height: int) -> float | None:
    if content_height <= 0 or (widget_top >= view_top and widget_bottom <= view_bottom):
        return None
    target = widget_top if widget_top < view_top else widget_bottom - (view_bottom - view_top)
    return max(0.0, min(1.0, target / content_height))


def model_row_label(model: dict, engines, memory_ceiling_gb: float | None) -> str:
    """One model row: per-engine sizes, plus a warning per engine it won't fit."""
    report = engine_fit_report(model, engines, memory_ceiling_gb)
    if not report:  # image checkpoints carry no per-engine weights
        return f"{model['label']}  {model.get('download_size', '')}".rstrip()
    label = f"{model['label']}  {format_engine_sizes(report)}"
    for warning in engine_fit_warnings(report, memory_ceiling_gb):
        label += f"   ⚠ {warning}"
    return label


def default_model_selection(memory_ceiling_gb: float | None,
                            engines=(LLAMACPP,)) -> dict[str, bool]:
    """Memory-aware defaults, matching terminal setup. Checked if it fits any engine."""
    selected: dict[str, bool] = {}
    for _, models in LLM_GROUPS:
        for model in models:
            selected[model["tag"]] = (not model.get("variant") or model.get("default", False)) and fits_any_engine(
                engine_fit_report(model, engines, memory_ceiling_gb),
            ) is not False
    for model in EMBED_MODELS:
        selected[model["tag"]] = True
    for model in IMAGE_MODELS:
        selected[model["short"]] = hardware.image_model_fits(
            model["checkpoint"], model["short"], memory_ceiling_gb,
        ) is not False
    return selected


def validate_gui_plan(plan: dict) -> list[str]:
    """Return user-facing validation errors for a completed wizard plan."""
    errors = []
    # Where image checkpoints come from is moot when none were picked.
    if plan.get("image_shorts") and plan.get("comfyui_mode") == "existing":
        entered = str(plan.get("comfyui_path", "")).strip()
        if not entered or not normalize_comfyui_dir(Path(entered)):
            errors.append("The existing ComfyUI path is not usable.")
    if "engines" in plan and not plan["engines"]:
        errors.append("Select at least one inference engine.")
    return errors


def next_page_index(current: int, step: int, enabled: list[bool]) -> int:
    """Nearest page in the `step` direction that applies, or `current` when there is none."""
    index = current + step
    while 0 <= index < len(enabled):
        if enabled[index]:
            return index
        index += step
    return current


def hf_token_review_label(plan: dict) -> str:
    """Describe the credential source without exposing the token."""
    if plan.get("hf_token"):
        return "provided and saved" if plan.get("save_hf_token") else "provided for this run only"
    if plan.get("use_existing_hf_token"):
        return "using existing token from HF_TOKEN or hf.txt"
    return "not provided"


def license_button_label(url: str) -> str:
    return "Review license…"


def selected_gui_token(existing_available: bool, override: bool, entered: str) -> str:
    """Use entered credentials only when replacement is available and enabled."""
    if existing_available and not override:
        return ""
    return entered.strip()


def should_save_gui_token(token: str, requested: bool) -> bool:
    return bool(token and requested)


def token_controls_enabled(existing_available: bool, override: bool) -> bool:
    return not existing_available or override


def sudo_notice(engines, package: str | None) -> str:
    """Warning shown before any privileged install, or "" when none will run."""
    if not package or "vllm" not in (engines or []):
        return ""
    return (f"Installing {package} needs administrator rights. You may be prompted for "
            "your password in the terminal window behind this wizard.")


def engine_checkbox_label(entry: dict) -> str:
    """Checkbox text for one engine row, including why a disabled one is unavailable."""
    label = entry["label"]
    if entry.get("experimental") and entry["enabled"]:
        label += " (experimental)"
    if not entry["enabled"]:
        label += " (unavailable on this system)"
    return f"{label} — {entry['note']}"


def build_setup_plan(*, model_selection: dict[str, bool], cleanup_names: list[str],
                     cleanup_selected: bool, vllm_cleanup_selection: dict[str, bool],
                     existing_hf_token: bool, override_token: bool, entered_token: str,
                     save_token: bool, comfyui_mode: str, comfyui_path: str,
                     engine_entries: list[dict], engine_selection: dict[str, bool]) -> dict:
    hf_token = selected_gui_token(existing_hf_token, override_token, entered_token)
    engines = [entry["name"] for entry in engine_entries
               if entry["enabled"] and engine_selection.get(entry["name"], False)]
    llm_models = [model for _, group in LLM_GROUPS for model in group]
    selected_llm = {model["tag"] for model in llm_models
                    if model_selection.get(model["tag"], False)}
    if VLLM in engines:
        selected_llm = collapse_variant_selection(llm_models, selected_llm)
    return {
        "llm_tags": [model["tag"] for model in llm_models if model["tag"] in selected_llm],
        "embedding_tags": [model["tag"] for model in EMBED_MODELS
                           if model_selection.get(model["tag"], False)],
        "image_shorts": [model["short"] for model in IMAGE_MODELS
                         if model_selection.get(model["short"], False)],
        "cleanup_names": cleanup_names if cleanup_selected else [],
        "vllm_cleanup_names": [name for name, selected in vllm_cleanup_selection.items()
                               if selected],
        "hf_token": hf_token,
        "save_hf_token": should_save_gui_token(hf_token, save_token),
        "use_existing_hf_token": existing_hf_token and not hf_token,
        "comfyui_mode": comfyui_mode,
        "comfyui_path": comfyui_path.strip(),
        "engines": engines,
    }


def setup_review_lines(plan: dict, *, show_engines: bool,
                       sudo_package: str | None) -> list[str]:
    lines = [
        f"LLM models: {len(plan['llm_tags'])}",
        f"Embedding models: {len(plan['embedding_tags'])}",
        f"Image models: {len(plan['image_shorts'])}",
        f"Delete non-catalog folders: {len(plan['cleanup_names'])}",
        f"Delete cached vLLM weights: {len(plan['vllm_cleanup_names'])}",
        f"Hugging Face token: {hf_token_review_label(plan)}",
    ]
    if plan["image_shorts"]:
        lines.append(f"ComfyUI: {plan['comfyui_mode']}")
    if show_engines:
        lines.append(f"Engines: {', '.join(plan['engines']) or 'none selected'}")
    notice = sudo_notice(plan["engines"], sudo_package)
    if notice:
        lines.extend(["", notice])
    if plan["image_shorts"] and plan["comfyui_path"]:
        lines.append(f"ComfyUI path: {plan['comfyui_path']}")
    return [*lines, "", "Nothing will be downloaded until you click Install."]


def run_setup_wizard_process(*, memory_ceiling_gb: float | None,
                             detected_comfyui: Path | None,
                             cleanup_names: list[str],
                             vllm_cleanup: list[dict] | None = None,
                             existing_hf_token: bool = False,
                             engine_entries: list[dict] | None = None,
                             sudo_package: str | None = None) -> dict | None:
    request_handle, request_name = tempfile.mkstemp(prefix="local-ai-bench-setup-request-", suffix=".json")
    response_handle, response_name = tempfile.mkstemp(prefix="local-ai-bench-setup-response-", suffix=".json")
    os.close(request_handle)
    os.close(response_handle)
    request_path, response_path = Path(request_name), Path(response_name)
    try:
        request_path.write_text(json.dumps({
            "memory_ceiling_gb": memory_ceiling_gb,
            "detected_comfyui": str(detected_comfyui) if detected_comfyui else None,
            "cleanup_names": cleanup_names,
            "vllm_cleanup": vllm_cleanup or [],
            "existing_hf_token": existing_hf_token,
            "engine_entries": engine_entries or [],
            "sudo_package": sudo_package,
        }))
        result = subprocess.run([
            sys.executable, "-m", "scripts.setup.setup_gui",
            "--request", str(request_path), "--response", str(response_path),
        ])
        if result.returncode != 0:
            raise RuntimeError("The graphical setup wizard stopped unexpectedly.")
        response = json.loads(response_path.read_text())
        return response.get("plan")
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def run_setup_wizard(*, memory_ceiling_gb: float | None,
                     detected_comfyui: Path | None,
                     cleanup_names: list[str],
                     vllm_cleanup: list[dict] | None = None,
                     existing_hf_token: bool = False,
                     engine_entries: list[dict] | None = None,
                     sudo_package: str | None = None) -> dict | None:  # pragma: no cover — interactive desktop UI
    import tkinter as tk
    import webbrowser
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Local AI Bench Setup")
    root.geometry("820x680")
    root.minsize(720, 580)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    configure_keyboard_accessibility(root, ttk)

    def bring_to_front() -> None:
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.after(400, lambda: root.attributes("-topmost", False))

    root.after(150, bring_to_front)

    engine_entries = engine_entries or []
    initial_engines = [entry["name"] for entry in engine_entries
                       if entry["checked"] and entry["enabled"]] or [LLAMACPP]
    defaults = default_model_selection(memory_ceiling_gb, initial_engines)
    model_vars = {key: tk.BooleanVar(value=value) for key, value in defaults.items()}
    labelled_models: dict[str, tuple] = {}
    variant_groups: dict[str, list[dict]] = {}
    for _, models in LLM_GROUPS:
        for model in models:
            if model.get("base_model") and model.get("variant"):
                variant_groups.setdefault(model["base_model"], []).append(model)
    variant_parent_widgets: dict[str, tuple] = {}
    variant_child_rows: dict[str, tuple] = {}
    applied_engines = list(initial_engines)
    token_var = tk.StringVar()
    save_token_var = tk.BooleanVar(value=True)
    override_token_var = tk.BooleanVar(value=False)
    cleanup_var = tk.BooleanVar(value=False)
    vllm_cleanup = list(vllm_cleanup or [])
    # One variable per cached repo: the vLLM cache is shared with other tools, so
    # each entry is opted into individually rather than as a group.
    vllm_cleanup_vars = {entry["directory_name"]: tk.BooleanVar(value=False)
                         for entry in vllm_cleanup}
    comfy_mode_var = tk.StringVar(value="detected" if detected_comfyui else "download")
    comfy_path_var = tk.StringVar(value=str(detected_comfyui or ""))
    result: dict | None = None
    pages: list[ttk.Frame] = []
    page_index = 0

    shell = ttk.Frame(root, padding=20)
    shell.grid(sticky="nsew")
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(1, weight=1)
    ttk.Label(shell, text="Local AI Bench Setup", font=("TkDefaultFont", 20, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 14),
    )
    content = ttk.Frame(shell)
    content.grid(row=1, column=0, sticky="nsew")
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)

    def new_page() -> ttk.Frame:
        page = ttk.Frame(content)
        page.grid(row=0, column=0, sticky="nsew")
        page.columnconfigure(0, weight=1)
        pages.append(page)
        return page

    welcome = new_page()
    ttk.Label(welcome, text="Welcome", font=("TkDefaultFont", 16, "bold")).grid(sticky="w")
    ttk.Label(
        welcome,
        text=("This wizard detects existing tools, lets you choose every model and credential option, "
              "and shows a final review before downloading anything."),
        wraplength=740, justify="left",
    ).grid(sticky="w", pady=(14, 8))
    memory_text = (f"Detected model-memory ceiling: approximately {memory_ceiling_gb:.1f} GB."
                   if memory_ceiling_gb is not None else
                   "A reliable model-memory ceiling could not be detected; all models start selected.")
    ttk.Label(welcome, text=memory_text, wraplength=740).grid(sticky="w", pady=8)
    if detected_comfyui:
        ttk.Label(welcome, text=f"Existing ComfyUI detected: {detected_comfyui}", wraplength=740).grid(sticky="w")
    engine_vars: dict[str, "tk.BooleanVar"] = {}
    if engine_entries:
        ttk.Label(welcome, text="Engines", font=("TkDefaultFont", 12, "bold")).grid(
            sticky="w", pady=(16, 0))
        ttk.Label(welcome, wraplength=740, justify="left",
                  text="Models you select later are downloaded for every engine checked here.",
                  ).grid(sticky="w")
        for entry in engine_entries:
            var = tk.BooleanVar(value=entry["checked"] and entry["enabled"])
            engine_vars[entry["name"]] = var
            ttk.Checkbutton(
                welcome, text=engine_checkbox_label(entry), variable=var,
                state="normal" if entry["enabled"] else "disabled",
            ).grid(sticky="w", pady=(4, 0))
        if sudo_package:
            ttk.Label(welcome, wraplength=740, justify="left",
                      text=("Note: selecting vLLM also installs "
                            f"{sudo_package}, which needs administrator rights. "
                            "You may be prompted for your password in the terminal."),
                      ).grid(sticky="w", pady=(8, 0))

    models_page = new_page()
    ttk.Label(models_page, text="Choose models", font=("TkDefaultFont", 16, "bold")).grid(sticky="w")
    ttk.Label(
        models_page,
        text="Models estimated not to fit are unchecked by default. You can still select them.",
    ).grid(sticky="w", pady=(4, 10))
    canvas = tk.Canvas(models_page, highlightthickness=0)
    scrollbar = ttk.Scrollbar(models_page, orient="vertical", command=canvas.yview)
    model_list = ttk.Frame(canvas)
    model_list.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    model_window = canvas.create_window((0, 0), window=model_list, anchor="nw")
    canvas.bind(
        "<Configure>", lambda event: canvas.itemconfigure(model_window, width=event.width),
    )
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")
    models_page.rowconfigure(2, weight=1)
    model_list.columnconfigure(0, weight=1)

    def selected_model_tags() -> set[str]:
        return {key for key, variable in model_vars.items() if variable.get()}

    def sync_variant_parents() -> None:
        selected = selected_model_tags()
        for base_model, (widget, _row) in variant_parent_widgets.items():
            state = variant_selection_state(
                [model["tag"] for model in variant_groups[base_model]], selected,
            )
            widget.state(["selected" if state == "all" else "!selected"])
            widget.state(["alternate" if state == "some" else "!alternate"])

    def toggle_variant_parent(base_model: str) -> None:
        tags = [model["tag"] for model in variant_groups[base_model]]
        target = variant_selection_target(tags, selected_model_tags())
        for tag in tags:
            model_vars[tag].set(tag in target)
        sync_variant_parents()

    def scroll_models(event):
        widget = root.winfo_containing(root.winfo_pointerx(), root.winfo_pointery())
        current = widget
        while current is not None and current not in {canvas, model_list}:
            current = getattr(current, "master", None)
        if current is None:
            return None
        units = mousewheel_scroll_units(
            delta=getattr(event, "delta", 0), button=getattr(event, "num", 0),
            platform_name=root.tk.call("tk", "windowingsystem"),
        )
        if units:
            canvas.yview_scroll(units, "units")
        return "break"

    root.bind_all("<MouseWheel>", scroll_models)
    root.bind_all("<Button-4>", scroll_models)
    root.bind_all("<Button-5>", scroll_models)

    row = 0
    rendered_variant_parents = set()
    for label, models in (*LLM_GROUPS, ("Embeddings", EMBED_MODELS), ("Image generation", IMAGE_MODELS)):
        ttk.Label(model_list, text=label, font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, sticky="w", pady=(9, 2),
        )
        row += 1
        for model in models:
            key = model.get("tag") or model["short"]
            base_model = model.get("base_model")
            if base_model and model.get("variant") and base_model not in rendered_variant_parents:
                parent = ttk.Checkbutton(
                    model_list, text=model.get("base_label", base_model),
                    command=lambda base=base_model: toggle_variant_parent(base),
                )
                parent.grid(row=row, column=0, sticky="w", padx=(16, 12), pady=(5, 1))
                variant_parent_widgets[base_model] = (parent, row)
                rendered_variant_parents.add(base_model)
                row += 1
            option_row = ttk.Frame(model_list)
            indent = 40 if base_model else 16
            option_row.grid(row=row, column=0, sticky="ew", padx=(indent, 12), pady=2)
            option_row.columnconfigure(1, weight=1)
            checkbutton = (
                ttk.Checkbutton(
                    option_row, text=model_row_label(model, initial_engines, memory_ceiling_gb),
                    variable=model_vars[key], command=sync_variant_parents,
                )
                if base_model else ttk.Checkbutton(
                    option_row, text=model_row_label(model, initial_engines, memory_ceiling_gb),
                    variable=model_vars[key],
                )
            )
            checkbutton.grid(row=0, column=0, columnspan=2, sticky="nw")
            if "download_size" in model:
                labelled_models[key] = (checkbutton, model)
            if base_model:
                variant_child_rows[key] = (option_row, row)
            license_url = model.get("license_url")
            if license_url:
                ttk.Button(
                    model_list, text=license_button_label(license_url),
                    command=lambda url=license_url: webbrowser.open(url),
                ).grid(row=row, column=1, sticky="e", pady=2)
            row += 1
    sync_variant_parents()

    def apply_variant_engine_mode(engines: list[str]) -> None:
        llamacpp_only = engines == [LLAMACPP]
        if not llamacpp_only:
            collapsed = collapse_variant_selection(
                [model for variants in variant_groups.values() for model in variants],
                selected_model_tags(),
            )
            for variants in variant_groups.values():
                for model in variants:
                    model_vars[model["tag"]].set(model["tag"] in collapsed)
        for widget, parent_row in variant_parent_widgets.values():
            if llamacpp_only:
                widget.grid(row=parent_row, column=0, sticky="w", padx=(16, 12), pady=(5, 1))
            else:
                widget.grid_remove()
        for variants in variant_groups.values():
            for model in variants:
                option_row, child_row = variant_child_rows[model["tag"]]
                label_widget, _ = labelled_models[model["tag"]]
                if llamacpp_only:
                    option_row.grid(row=child_row, column=0, sticky="ew", padx=(40, 12), pady=2)
                    child_model = {**model, "label": model["variant"]}
                    label_widget.configure(text=model_row_label(child_model, engines, memory_ceiling_gb))
                elif model.get("default"):
                    option_row.grid(row=child_row, column=0, sticky="ew", padx=(16, 12), pady=2)
                    base_entry = {**model, "label": model.get("base_label", model["label"])}
                    label_widget.configure(text=model_row_label(base_entry, engines, memory_ceiling_gb))
                else:
                    option_row.grid_remove()
        sync_variant_parents()

    apply_variant_engine_mode(initial_engines)
    if cleanup_names:
        ttk.Label(
            model_list, font=("TkDefaultFont", 11, "bold"),
            text="Downloaded llama.cpp models not in the catalog",
        ).grid(row=row, column=0, sticky="w", pady=(14, 0))
        row += 1
        ttk.Label(
            model_list, wraplength=520,
            text=("These folders are in this project's own models directory, usually left "
                  "behind by an earlier catalog. Nothing outside it is touched."),
        ).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        ttk.Checkbutton(
            model_list,
            text=f"Delete {len(cleanup_names)} model folder(s): {', '.join(cleanup_names)}",
            variable=cleanup_var,
        ).grid(row=row, column=0, sticky="w")
        row += 1
    if vllm_cleanup:
        ttk.Label(
            model_list, font=("TkDefaultFont", 11, "bold"),
            text="Cached vLLM weights not in the catalog",
        ).grid(row=row, column=0, sticky="w", pady=(14, 0))
        row += 1
        ttk.Label(
            model_list, wraplength=520,
            text=("This cache is shared with anything else on this machine that uses "
                  "Hugging Face. Delete only what you recognize."),
        ).grid(row=row, column=0, sticky="w", pady=(0, 4))
        row += 1
        for entry in vllm_cleanup:
            ttk.Checkbutton(
                model_list,
                text=f"Delete {entry['repo']}  (~{entry['size'] / 1e9:.1f} GB)",
                variable=vllm_cleanup_vars[entry["directory_name"]],
            ).grid(row=row, column=0, sticky="w")
            row += 1

    credentials = new_page()
    ttk.Label(credentials, text="Hugging Face", font=("TkDefaultFont", 16, "bold")).grid(sticky="w")
    ttk.Label(
        credentials,
        text=("A token is optional for public models and required for selected gated image models. "
              "It is never written to the setup configuration."),
        wraplength=740,
    ).grid(sticky="w", pady=(6, 14))
    override_token_check = None
    if existing_hf_token:
        ttk.Label(
            credentials, text="A token is already available from HF_TOKEN or hf.txt.",
        ).grid(sticky="w", pady=(0, 10))
        override_token_check = ttk.Checkbutton(
            credentials, text="Override token", variable=override_token_var,
        )
        override_token_check.grid(sticky="w", pady=(0, 10))
    ttk.Label(credentials, text="Access token").grid(sticky="w")
    token_entry = ttk.Entry(credentials, textvariable=token_var, show="•", width=72)
    token_entry.grid(sticky="ew", pady=(3, 10))
    save_token_check = ttk.Checkbutton(
        credentials, text="Save token to gitignored hf.txt for future runs",
        variable=save_token_var,
    )
    save_token_check.grid(sticky="w")

    token_help = ttk.Frame(credentials)
    ttk.Button(
        token_help, text="Open Hugging Face login",
        command=lambda: webbrowser.open(HF_LOGIN_URL),
    ).grid(sticky="w", pady=(0, 8))
    ttk.Label(
        token_help,
        text=("1. Login\n"
              "2. Click your avatar in the upper right.\n"
              "3. Choose Settings from the menu that appears.\n"
              "4. Go to Access Tokens.\n"
              "5. Create a new token.\n"
              "6. Copy and paste it into the Access token field."),
        justify="left",
    ).grid(sticky="w")

    def update_token_controls() -> None:
        enabled = token_controls_enabled(existing_hf_token, override_token_var.get())
        state = "normal" if enabled else "disabled"
        token_entry.configure(state=state)
        save_token_check.configure(state=state)
        if enabled:
            token_help.grid(sticky="w", pady=(14, 0))
        else:
            token_help.grid_remove()
        refresh_tk_layout(credentials)

    if override_token_check is not None:
        override_token_check.configure(command=update_token_controls)
    update_token_controls()

    comfy = new_page()
    ttk.Label(comfy, text="ComfyUI", font=("TkDefaultFont", 16, "bold")).grid(sticky="w")
    ttk.Label(
        comfy,
        text="Image models stay in Local AI Bench's models/comfyui directory in every mode.",
        wraplength=740,
    ).grid(sticky="w", pady=(6, 12))
    if detected_comfyui:
        ttk.Radiobutton(
            comfy, text=f"Use detected installation: {detected_comfyui}",
            variable=comfy_mode_var, value="detected",
        ).grid(sticky="w", pady=3)
    ttk.Radiobutton(
        comfy, text="Download a managed ComfyUI copy if image models are selected",
        variable=comfy_mode_var, value="download",
    ).grid(sticky="w", pady=3)
    ttk.Radiobutton(
        comfy, text="Use another existing installation",
        variable=comfy_mode_var, value="existing",
    ).grid(sticky="w", pady=3)
    path_row = ttk.Frame(comfy)
    path_row.grid(sticky="ew", padx=(24, 0), pady=(8, 0))
    path_row.columnconfigure(0, weight=1)
    ttk.Entry(path_row, textvariable=comfy_path_var).grid(row=0, column=0, sticky="ew")
    ttk.Button(
        path_row, text="Browse…",
        command=lambda: comfy_path_var.set(filedialog.askdirectory() or comfy_path_var.get()),
    ).grid(row=0, column=1, padx=(8, 0))

    review = new_page()
    ttk.Label(review, text="Review", font=("TkDefaultFont", 16, "bold")).grid(sticky="w")
    review_text = tk.Text(review, height=24, wrap="word", state="disabled")
    review_text.grid(sticky="nsew", pady=(10, 0))
    review.rowconfigure(1, weight=1)

    nav = ttk.Frame(shell)
    nav.grid(row=2, column=0, sticky="ew", pady=(16, 0))
    back_button = ttk.Button(nav, text="Back")
    back_button.pack(side="left")
    cancel_button = ttk.Button(nav, text="Cancel")
    cancel_button.pack(side="right")
    next_button = ttk.Button(nav, text="Next")
    next_button.pack(side="right", padx=(0, 8))

    def build_plan() -> dict:
        return build_setup_plan(
            model_selection={name: variable.get() for name, variable in model_vars.items()},
            cleanup_names=cleanup_names, cleanup_selected=cleanup_var.get(),
            vllm_cleanup_selection={
                name: variable.get() for name, variable in vllm_cleanup_vars.items()
            },
            existing_hf_token=existing_hf_token, override_token=override_token_var.get(),
            entered_token=token_var.get(), save_token=save_token_var.get(),
            comfyui_mode=comfy_mode_var.get(), comfyui_path=comfy_path_var.get(),
            engine_entries=engine_entries,
            engine_selection={name: variable.get() for name, variable in engine_vars.items()},
        )

    def refresh_review() -> None:
        plan = build_plan()
        lines = setup_review_lines(
            plan, show_engines=bool(engine_entries), sudo_package=sudo_package,
        )
        review_text.configure(state="normal")
        review_text.delete("1.0", "end")
        review_text.insert("1.0", "\n".join(lines))
        review_text.configure(state="disabled")

    def selected_engines() -> list[str]:
        chosen = [entry["name"] for entry in engine_entries
                  if entry["enabled"] and engine_vars[entry["name"]].get()]
        return chosen or [LLAMACPP]

    def refresh_model_rows() -> None:
        """Re-label and re-default the model list for the checked engines."""
        nonlocal applied_engines
        engines = selected_engines()
        if engines == applied_engines:
            return
        applied_engines = engines
        for key, (checkbutton, model) in labelled_models.items():
            checkbutton.configure(text=model_row_label(model, engines, memory_ceiling_gb))
        for key, value in default_model_selection(memory_ceiling_gb, engines).items():
            if key in model_vars:
                model_vars[key].set(value)
        apply_variant_engine_mode(engines)

    def show_page(index: int) -> None:
        nonlocal page_index
        page_index = index
        if pages[index] is models_page:
            refresh_model_rows()
        display_wizard_page(pages, index)
        back_button.configure(state="disabled" if index == 0 else "normal")
        next_button.configure(text="Install" if index == len(pages) - 1 else "Next")
        if index == len(pages) - 1:
            refresh_review()
        refresh_tk_layout(root)
        controls = []

        def collect_controls(widget) -> None:
            for child in widget.winfo_children():
                if isinstance(child, (
                    ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Entry, ttk.Combobox,
                    tk.Text,
                )):
                    controls.append(child)
                collect_controls(child)

        collect_controls(pages[index])
        page_controls = sorted(
            controls,
            key=lambda widget: (widget.winfo_rooty(), widget.winfo_rootx()),
        )
        configure_explicit_tab_order([
            *page_controls, back_button, next_button, cancel_button,
        ])
        if pages[index] is models_page:
            def reveal_model_control(event) -> None:
                region = canvas.bbox("all")
                if region is None:
                    return
                widget_top = event.widget.winfo_rooty() - model_list.winfo_rooty()
                fraction = focus_scroll_fraction(
                    widget_top=widget_top,
                    widget_bottom=widget_top + event.widget.winfo_height(),
                    view_top=int(canvas.canvasy(0)),
                    view_bottom=int(canvas.canvasy(canvas.winfo_height())),
                    content_height=region[3] - region[1],
                )
                if fraction is not None:
                    canvas.yview_moveto(fraction)

            for control in page_controls:
                control.bind("<FocusIn>", reveal_model_control, add="+")

    def page_enabled() -> list[bool]:
        image_selected = any(model_vars[model["short"]].get() for model in IMAGE_MODELS)
        return [page is not comfy or image_selected for page in pages]

    def go_back() -> None:
        show_page(next_page_index(page_index, -1, page_enabled()))

    def go_next() -> None:
        nonlocal result
        if page_index < len(pages) - 1:
            if pages[page_index] is comfy:
                errors = validate_gui_plan(build_plan())
                if errors:
                    messagebox.showerror("Check ComfyUI", "\n".join(errors))
                    return
            show_page(next_page_index(page_index, 1, page_enabled()))
            return
        plan = build_plan()
        errors = validate_gui_plan(plan)
        if errors:
            messagebox.showerror("Cannot start setup", "\n".join(errors))
            return
        result = plan
        root.withdraw()
        root.after(150, root.quit)

    back_button.configure(command=go_back)
    cancel_button.configure(command=root.quit)
    next_button.configure(command=go_next)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    show_page(0)
    root.mainloop()
    root.destroy()
    return result


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    plan = run_setup_wizard(
        memory_ceiling_gb=request["memory_ceiling_gb"],
        detected_comfyui=Path(request["detected_comfyui"]) if request["detected_comfyui"] else None,
        cleanup_names=request["cleanup_names"],
        vllm_cleanup=request.get("vllm_cleanup", []),
        existing_hf_token=request["existing_hf_token"],
        engine_entries=request.get("engine_entries") or [],
        sudo_package=request.get("sudo_package"),
    )
    args.response.write_text(json.dumps({"plan": plan}))


if __name__ == "__main__":  # pragma: no cover
    main()
