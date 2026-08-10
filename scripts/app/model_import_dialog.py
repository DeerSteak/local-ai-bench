"""Tk dialog for inspecting and importing a Hugging Face custom model."""

import threading
from pathlib import Path

from scripts.runtime import config
from scripts.setup.custom_models import custom_model
from scripts.setup.model_download import enough_disk_space, import_model, load_hf_token
from scripts.setup.model_import import default_custom_tag, inspect_repository, valid_custom_tag
from scripts.workloads.models import EMBED_MODELS, LLM_MODELS


def show_model_import_dialog(*, root, tk, ttk, messagebox, available_engines,
                             engine_factory, on_imported) -> None:  # pragma: no cover — interactive Tk UI
    engines = [
        name for name in available_engines if name in {"llamacpp", "vllm"}
        and getattr(engine_factory(name), "supports_model_import", lambda: True)()
    ]
    if not engines:
        messagebox.showerror(
            "Import unavailable", "Install a locally managed llama.cpp or vLLM first.", parent=root,
        )
        return

    dialog = tk.Toplevel(root)
    dialog.title("Import Hugging Face Model")
    dialog.geometry("700x650")
    dialog.minsize(620, 560)
    dialog.transient(root)
    dialog.grab_set()
    shell = ttk.Frame(dialog, padding=18)
    shell.pack(fill="both", expand=True)
    shell.columnconfigure(1, weight=1)

    variables = {
        "repo": tk.StringVar(), "revision": tk.StringVar(value="main"),
        "engine": tk.StringVar(), "variant": tk.StringVar(),
        "label": tk.StringVar(), "tag": tk.StringVar(),
        "acknowledge": tk.BooleanVar(value=False),
        "support": tk.StringVar(value="Paste a Hugging Face model repository, then inspect it."),
        "validation": tk.StringVar(),
        "destination": tk.StringVar(value="Destination will be shown after inspection."),
    }
    state = {"inspection": None, "variants": {}}

    ttk.Label(shell, text="Import Hugging Face Model", style="Title.TLabel").grid(
        row=0, column=0, columnspan=3, sticky="w",
    )
    ttk.Label(shell, text="Imports full model weights as an engine-specific Custom LLM.").grid(
        row=1, column=0, columnspan=3, sticky="w", pady=(2, 14),
    )
    ttk.Label(shell, text="Repository").grid(row=2, column=0, sticky="w")
    repo_entry = ttk.Entry(shell, textvariable=variables["repo"])
    repo_entry.grid(row=2, column=1, sticky="ew", padx=(10, 8))
    inspect_button = ttk.Button(shell, text="Inspect Repo")
    inspect_button.grid(row=2, column=2)
    ttk.Label(shell, text="Revision").grid(row=3, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(shell, textvariable=variables["revision"]).grid(
        row=3, column=1, sticky="ew", padx=(10, 8), pady=(8, 0),
    )
    ttk.Label(shell, textvariable=variables["support"], wraplength=640).grid(
        row=4, column=0, columnspan=3, sticky="w", pady=12,
    )
    ttk.Separator(shell).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 12))
    ttk.Label(shell, text="Engine").grid(row=6, column=0, sticky="w")
    engine_combo = ttk.Combobox(shell, textvariable=variables["engine"], state="disabled")
    engine_combo.grid(row=6, column=1, sticky="w", padx=(10, 0))
    ttk.Label(shell, text="Variant").grid(row=7, column=0, sticky="w", pady=(8, 0))
    variant_combo = ttk.Combobox(shell, textvariable=variables["variant"], state="disabled")
    variant_combo.grid(row=7, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(8, 0))
    for row, (label, key) in enumerate((("Display name", "label"), ("Model tag", "tag")), 8):
        ttk.Label(shell, text=label).grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(shell, textvariable=variables[key]).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(10, 0),
        )
    ttk.Label(shell, textvariable=variables["destination"], wraplength=640).grid(
        row=10, column=0, columnspan=3, sticky="w", pady=(12, 0),
    )
    acknowledgement = ttk.Checkbutton(
        shell, variable=variables["acknowledge"], state="disabled",
        text="Import despite unverified runtime compatibility",
    )
    acknowledgement.grid(row=11, column=0, columnspan=3, sticky="w", pady=(14, 0))
    ttk.Label(shell, textvariable=variables["validation"], wraplength=640).grid(
        row=12, column=0, columnspan=3, sticky="w", pady=(8, 0),
    )
    progress = ttk.Progressbar(shell, mode="indeterminate")
    progress.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(14, 0))
    actions = ttk.Frame(shell)
    actions.grid(row=14, column=0, columnspan=3, sticky="e", pady=(18, 0))
    ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="left")
    import_button = ttk.Button(actions, text="Import Model", state="disabled")
    import_button.pack(side="left", padx=(10, 0))

    def size_label(size):
        return "size unknown" if size is None else f"{size / (1024 ** 3):.1f} GiB"

    def selected_variant():
        return state["variants"].get(variables["variant"].get())

    def destination(engine, tag):
        return (config.MODELS_DIR / "llamacpp" / (tag or "<tag>") if engine == "llamacpp"
                else getattr(engine_factory("vllm"), "cache_home")())

    def validate(*_args):
        inspection, engine, variant = state["inspection"], variables["engine"].get(), selected_variant()
        tag, label = variables["tag"].get(), variables["label"].get().strip()
        reason = ""
        if inspection is None:
            reason = "Inspect a repository before importing."
        elif engine not in engines or variant is None:
            reason = "Select an available engine and artifact variant."
        elif not valid_custom_tag(tag):
            reason = "Tag must use only letters, numbers, dots, underscores, or hyphens."
        elif not label:
            reason = "Display name is required."
        elif tag in {model["tag"] for model in LLM_MODELS + EMBED_MODELS}:
            reason = "That tag belongs to a catalog model."
        elif custom_model(engine, tag) is not None:
            reason = "That custom tag is already registered for this engine."
        elif not variables["acknowledge"].get():
            reason = "Acknowledge that runtime compatibility is unverified."
        elif enough_disk_space(variant, destination(engine, tag)) is False:
            reason = "Not enough free disk space for this variant."
        variables["validation"].set(reason or "Ready to import.")
        import_button.configure(state="disabled" if reason else "normal")

    def refresh_variants(*_args):
        inspection, engine = state["inspection"], variables["engine"].get()
        if inspection is None:
            return
        variants = (inspection.llama_variants if engine == "llamacpp" else
                    (inspection.vllm_variant,) if inspection.vllm_variant else ())
        choices = {f"{item.label} — {size_label(item.size)}": item for item in variants}
        state["variants"] = choices
        variant_combo.configure(values=tuple(choices), state="readonly" if choices else "disabled")
        variables["variant"].set(next(iter(choices), ""))
        variables["destination"].set(f"Destination: {destination(engine, variables['tag'].get())}")
        validate()

    def inspection_finished(result=None, error=None):
        progress.stop()
        inspect_button.configure(state="normal")
        if error is not None:
            state["inspection"] = None
            variables["support"].set(f"Repository inspection failed: {error}")
            engine_combo.configure(state="disabled", values=())
            validate()
            return
        assert result is not None
        state["inspection"] = result
        supported = []
        if result.llama_variants and "llamacpp" in engines:
            supported.append("llamacpp")
        if result.vllm_variant and "vllm" in engines:
            supported.append("vllm")
        variables["support"].set(
            f"llama.cpp: {len(result.llama_variants)} GGUF variant(s) · "
            + ("vLLM: safetensors snapshot" if result.vllm_variant else "vLLM: unavailable")
        )
        engine_combo.configure(values=supported, state="readonly" if supported else "disabled")
        variables["engine"].set(supported[0] if supported else "")
        variables["label"].set(result.repo.rsplit("/", 1)[-1])
        variables["tag"].set(default_custom_tag(result.repo))
        variables["revision"].set(result.revision)
        acknowledgement.configure(state="normal" if supported else "disabled")
        variables["acknowledge"].set(False)
        refresh_variants()
        if not supported:
            variables["validation"].set("This repository cannot be imported by an installed engine.")

    def inspect_repo():
        inspect_button.configure(state="disabled")
        import_button.configure(state="disabled")
        progress.start(12)
        variables["support"].set("Inspecting repository metadata…")

        def worker():
            try:
                result = inspect_repository(
                    variables["repo"].get(), variables["revision"].get(), token=load_hf_token(),
                )
                root.after(0, lambda: inspection_finished(result=result))
            except Exception as exc:
                root.after(0, lambda error=exc: inspection_finished(error=error))
        threading.Thread(target=worker, daemon=True).start()

    def begin_import():
        inspection, variant, engine = state["inspection"], selected_variant(), variables["engine"].get()
        if inspection is None or variant is None:
            return
        import_button.configure(state="disabled")
        progress.start(12)
        variables["validation"].set("Downloading and validating model files…")

        def worker():
            try:
                import_model(
                    inspection=inspection, engine=engine, variant=variant,
                    tag=variables["tag"].get(), label=variables["label"].get(), token=load_hf_token(),
                    vllm_cache=(getattr(engine_factory("vllm"), "cache_home")()
                                if engine == "vllm" else None),
                )
                root.after(0, import_finished)
            except Exception as exc:
                root.after(0, lambda error=exc: import_failed(error))
        threading.Thread(target=worker, daemon=True).start()

    def import_finished():
        progress.stop()
        tag, label, engine = variables["tag"].get(), variables["label"].get(), variables["engine"].get()
        on_imported(tag)
        dialog.destroy()
        messagebox.showinfo("Model imported", f"{label} was added as {tag} for {engine}.", parent=root)

    def import_failed(error):
        progress.stop()
        inspect_button.configure(state="normal")
        variables["validation"].set(f"Import failed: {error}")
        validate()

    inspect_button.configure(command=inspect_repo)
    import_button.configure(command=begin_import)
    engine_combo.bind("<<ComboboxSelected>>", refresh_variants)
    for key in ("variant", "label", "tag", "acknowledge"):
        variables[key].trace_add("write", validate)
    repo_entry.focus_set()
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
