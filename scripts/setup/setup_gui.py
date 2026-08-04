"""Tkinter wizard that collects a complete setup plan before installation."""

from pathlib import Path

from scripts.runtime import hardware
from scripts.workloads.models import (
    EMBED_MODELS,
    IMAGE_MODELS,
    LLM_MODELS_LARGE,
    LLM_MODELS_MEDIUM,
    LLM_MODELS_SMALL,
    LLM_MODELS_XSMALL,
)
from scripts.runtime.comfyui_installation import normalize_comfyui_dir
from scripts.app.tk_utils import mousewheel_scroll_units


LLM_GROUPS = (
    ("Extra-small LLMs", LLM_MODELS_XSMALL),
    ("Small LLMs", LLM_MODELS_SMALL),
    ("Medium LLMs", LLM_MODELS_MEDIUM),
    ("Large LLMs", LLM_MODELS_LARGE),
)


def default_model_selection(memory_ceiling_gb: float | None) -> dict[str, bool]:
    """Return the same memory-aware defaults used by terminal setup."""
    selected: dict[str, bool] = {}
    for _, models in LLM_GROUPS:
        for model in models:
            selected[model["tag"]] = hardware.model_fits(
                model["download_size"], memory_ceiling_gb,
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
    if plan.get("comfyui_mode") == "existing":
        entered = str(plan.get("comfyui_path", "")).strip()
        if not entered or not normalize_comfyui_dir(Path(entered)):
            errors.append("The existing ComfyUI path is not usable.")
    return errors


def run_setup_wizard(*, memory_ceiling_gb: float | None,
                     detected_comfyui: Path | None,
                     cleanup_names: list[str],
                     existing_hf_token: bool = False) -> dict | None:  # pragma: no cover — interactive desktop UI
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Local AI Bench Setup")
    root.geometry("820x680")
    root.minsize(720, 580)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    def bring_to_front() -> None:
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()
        root.after(400, lambda: root.attributes("-topmost", False))

    root.after(150, bring_to_front)

    defaults = default_model_selection(memory_ceiling_gb)
    model_vars = {key: tk.BooleanVar(value=value) for key, value in defaults.items()}
    token_var = tk.StringVar()
    save_token_var = tk.BooleanVar(value=True)
    cleanup_var = tk.BooleanVar(value=False)
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
    canvas.create_window((0, 0), window=model_list, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")
    models_page.rowconfigure(2, weight=1)

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
    for label, models in (*LLM_GROUPS, ("Embeddings", EMBED_MODELS), ("Image generation", IMAGE_MODELS)):
        ttk.Label(model_list, text=label, font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, sticky="w", pady=(9, 2),
        )
        row += 1
        for model in models:
            key = model.get("tag", model.get("short"))
            size = model.get("download_size", "")
            ttk.Checkbutton(
                model_list, text=f"{model['label']}  {size}", variable=model_vars[key],
            ).grid(row=row, column=0, sticky="w", padx=(16, 0))
            row += 1
    if cleanup_names:
        ttk.Checkbutton(
            model_list,
            text=f"Delete {len(cleanup_names)} non-catalog model folder(s): {', '.join(cleanup_names)}",
            variable=cleanup_var,
        ).grid(row=row, column=0, sticky="w", pady=(14, 4))

    credentials = new_page()
    ttk.Label(credentials, text="Hugging Face", font=("TkDefaultFont", 16, "bold")).grid(sticky="w")
    ttk.Label(
        credentials,
        text=("A token is optional for public models and required for selected gated image models. "
              "It is never written to the setup configuration."),
        wraplength=740,
    ).grid(sticky="w", pady=(6, 14))
    if existing_hf_token:
        ttk.Label(
            credentials, text="A token is already available from HF_TOKEN or hf.txt.",
        ).grid(sticky="w", pady=(0, 10))
    ttk.Label(credentials, text="Access token").grid(sticky="w")
    ttk.Entry(credentials, textvariable=token_var, show="•", width=72).grid(sticky="ew", pady=(3, 10))
    ttk.Checkbutton(
        credentials, text="Save token to gitignored hf.txt for future runs",
        variable=save_token_var,
    ).grid(sticky="w")

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
    cancel_button = ttk.Button(nav, text="Cancel", command=root.destroy)
    cancel_button.pack(side="right")
    next_button = ttk.Button(nav, text="Next")
    next_button.pack(side="right", padx=(0, 8))

    def build_plan() -> dict:
        return {
            "llm_tags": [m["tag"] for _, group in LLM_GROUPS for m in group if model_vars[m["tag"]].get()],
            "embedding_tags": [m["tag"] for m in EMBED_MODELS if model_vars[m["tag"]].get()],
            "image_shorts": [m["short"] for m in IMAGE_MODELS if model_vars[m["short"]].get()],
            "cleanup_names": cleanup_names if cleanup_var.get() else [],
            "hf_token": token_var.get().strip(),
            "save_hf_token": save_token_var.get(),
            "use_existing_hf_token": existing_hf_token and not token_var.get().strip(),
            "comfyui_mode": comfy_mode_var.get(),
            "comfyui_path": comfy_path_var.get().strip(),
        }

    def refresh_review() -> None:
        plan = build_plan()
        lines = [
            f"LLM models: {len(plan['llm_tags'])}",
            f"Embedding models: {len(plan['embedding_tags'])}",
            f"Image models: {len(plan['image_shorts'])}",
            f"Delete non-catalog folders: {len(plan['cleanup_names'])}",
            f"Hugging Face token: {'provided and saved' if plan['hf_token'] and plan['save_hf_token'] else 'provided for this run only' if plan['hf_token'] else 'not provided'}",
            f"ComfyUI: {plan['comfyui_mode']}",
        ]
        if plan["comfyui_path"]:
            lines.append(f"ComfyUI path: {plan['comfyui_path']}")
        lines.extend(["", "Nothing will be downloaded until you click Install."])
        review_text.configure(state="normal")
        review_text.delete("1.0", "end")
        review_text.insert("1.0", "\n".join(lines))
        review_text.configure(state="disabled")

    def show_page(index: int) -> None:
        nonlocal page_index
        page_index = index
        pages[index].tkraise()
        back_button.configure(state="disabled" if index == 0 else "normal")
        next_button.configure(text="Install" if index == len(pages) - 1 else "Next")
        if index == len(pages) - 1:
            refresh_review()

    def go_back() -> None:
        if page_index:
            show_page(page_index - 1)

    def go_next() -> None:
        nonlocal result
        if page_index < len(pages) - 1:
            if pages[page_index] is comfy:
                errors = validate_gui_plan(build_plan())
                if errors:
                    messagebox.showerror("Check ComfyUI", "\n".join(errors))
                    return
            show_page(page_index + 1)
            return
        plan = build_plan()
        errors = validate_gui_plan(plan)
        if errors:
            messagebox.showerror("Cannot start setup", "\n".join(errors))
            return
        result = plan
        root.destroy()

    back_button.configure(command=go_back)
    next_button.configure(command=go_next)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    show_page(0)
    root.mainloop()
    return result
