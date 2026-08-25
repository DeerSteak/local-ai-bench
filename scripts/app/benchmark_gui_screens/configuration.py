"""Configuration screen layout and configuration-control ownership."""

from dataclasses import dataclass
from typing import Any

from scripts.app.benchmark_frontend import MAX_PROMPT_TOKEN_OPTIONS, TEST_DEFINITIONS, TG_TOKEN_OPTIONS
from scripts.app.benchmark_gui_support import plan_preview_sections
from scripts.runtime import config
from scripts.workloads.model_variants import variant_selection_state, variant_selection_target


@dataclass
class ConfigurationScreen:
    frame: Any
    canvas: Any
    form: Any
    configuration_frame: Any
    engine_box: Any
    preset_row: Any
    project_row: Any
    tests_box: Any
    models_box: Any
    model_rows: Any
    workload_box: Any
    advanced_toggle: Any
    test_widgets: dict
    test_labels: dict
    model_widgets: dict
    ttk: Any
    custom_models: list
    model_vars: dict
    model_defaults: dict
    tk: Any
    variant_children_visible: bool = True
    variant_parent_widgets: dict | None = None

    def variant_groups(self) -> dict[str, list]:
        groups: dict[str, list] = {}
        for entry in self.custom_models:
            if entry.base_model and entry.variant:
                groups.setdefault(entry.base_model, []).append(entry)
        return groups

    def sync_variant_parents(self) -> None:
        selected = {key for key, variable in self.model_vars.items() if variable.get()}
        for base_model, widget in (self.variant_parent_widgets or {}).items():
            state = variant_selection_state(
                [entry.value for entry in self.variant_groups()[base_model]], selected,
            )
            widget.state(["selected" if state == "all" else "!selected"])
            widget.state(["alternate" if state == "some" else "!alternate"])

    def toggle_variant_parent(self, base_model: str) -> None:
        groups = self.variant_groups()
        tags = [entry.value for entry in groups[base_model]]
        selected = {key for key, variable in self.model_vars.items() if variable.get()}
        target = variant_selection_target(tags, selected)
        for tag in tags:
            self.model_vars[tag].set(tag in target)
        self.sync_variant_parents()

    def set_variant_children_visible(self, visible: bool) -> None:
        if self.variant_children_visible != visible:
            self.variant_children_visible = visible
            self.render_model_rows()

    def render_model_rows(self) -> None:
        for child in self.model_rows.winfo_children():
            child.destroy()
        self.model_widgets.clear()
        self.variant_parent_widgets = {}
        variant_groups = self.variant_groups()
        rendered_parents = set()
        previous = None
        row = 0
        for entry in self.custom_models:
            if entry.base_model and entry.variant and not self.variant_children_visible \
                    and not entry.default_variant:
                continue
            if entry.section != previous:
                self.ttk.Label(
                    self.model_rows, text=entry.section, style="Section.TLabel",
                ).grid(row=row, column=0, sticky="w", pady=(7, 2))
                row += 1
                previous = entry.section
            if entry.base_model and entry.variant and self.variant_children_visible \
                    and entry.base_model not in rendered_parents:
                parent = self.ttk.Checkbutton(
                    self.model_rows, text=entry.base_label or entry.base_model,
                    command=lambda base=entry.base_model: self.toggle_variant_parent(base),
                )
                parent.grid(row=row, column=0, sticky="w", padx=(12, 0), pady=(4, 1))
                self.variant_parent_widgets[entry.base_model] = parent
                rendered_parents.add(entry.base_model)
                row += 1
            option_row = self.ttk.Frame(self.model_rows)
            indent = 32 if entry.base_model and self.variant_children_visible else 12
            option_row.grid(row=row, column=0, sticky="ew", padx=(indent, 0), pady=2)
            option_row.columnconfigure(1, weight=1)
            label_text = (
                entry.label.removeprefix(f"{entry.base_label} — ")
                if self.variant_children_visible and entry.base_model
                else entry.base_label or entry.label if entry.base_model
                else entry.label
            )
            widget = self.ttk.Checkbutton(
                option_row, text=label_text, variable=self.model_vars[entry.value],
                command=self.sync_variant_parents if entry.base_model else None,
            )
            widget.grid(row=0, column=0, columnspan=2, sticky="nw")
            self.ttk.Button(
                self.model_rows, text="Reset", width=6,
                command=lambda key=entry.value: self.model_vars[key].set(self.model_defaults[key]),
            ).grid(row=row, column=1, sticky="e", padx=(8, 0))
            self.model_widgets[entry.value] = widget
            row += 1
        self.sync_variant_parents()


def build_configuration_screen(notebook, *, tk, ttk, discovery: dict,
                               advanced_var, preset_var, project_status,
                               preset_names: list[str],
                               test_vars: dict, test_defaults: dict, custom_tests: list,
                               model_vars: dict, model_defaults: dict, custom_models: list,
                               cap_var, tg_vars: dict) -> ConfigurationScreen:
    frame = ttk.Frame(notebook, padding=18)
    notebook.add(frame, text="Configuration")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=1)
    ttk.Label(frame, text=f"Local AI Bench v{config.VERSION}", style="Title.TLabel").grid(
        row=0, column=0, sticky="w",
    )
    ttk.Label(
        frame, text="Choose a preset or adjust any setting to create a remembered Custom configuration.",
    ).grid(row=1, column=0, sticky="w", pady=(2, 12))
    canvas = tk.Canvas(frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    form = ttk.Frame(canvas)
    form.columnconfigure(0, weight=1)
    form.columnconfigure(1, weight=1)
    window_id = canvas.create_window((0, 0), window=form, anchor="nw")
    form.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")
    _build_discovery(ttk, form, discovery)
    configuration = ttk.Frame(form)
    configuration.grid(row=1, column=0, columnspan=2, sticky="nsew")
    configuration.columnconfigure(0, weight=1, uniform="configuration")
    configuration.columnconfigure(1, weight=1, uniform="configuration")
    header = ttk.Frame(configuration)
    header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    engine_box = ttk.LabelFrame(header, text="Inference engines", padding=12)
    engine_box.pack(side="top", fill="x", pady=(0, 10))
    preset_row = ttk.Frame(header)
    preset_row.pack(side="top", fill="x")
    ttk.Label(preset_row, text="Preset").pack(side="left")
    ttk.Combobox(
        preset_row, state="readonly", textvariable=preset_var,
        values=preset_names, width=24,
    ).pack(side="left", padx=(8, 8))
    project_row = ttk.Frame(configuration)
    project_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
    ttk.Label(project_row, textvariable=project_status).pack(side="left", padx=(0, 12))
    advanced_toggle = ttk.Checkbutton(
        configuration, text="Show advanced execution and path settings", variable=advanced_var,
    )
    advanced_toggle.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))
    tests_box, test_widgets, test_labels = _build_tests(
        ttk, configuration, custom_tests, test_vars, test_defaults,
    )
    models_box = ttk.LabelFrame(configuration, text="Installed models", padding=12)
    models_box.grid(row=3, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
    models_box.columnconfigure(0, weight=1)
    model_rows = ttk.Frame(models_box)
    model_rows.grid(row=0, column=0, columnspan=2, sticky="ew")
    model_rows.columnconfigure(0, weight=1)
    ttk.Label(
        models_box, text="Each checked model runs through every applicable selected workload.",
        wraplength=330,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
    workload_box = _build_workload(ttk, configuration, cap_var, tg_vars)
    screen = ConfigurationScreen(
        frame, canvas, form, configuration, engine_box, preset_row, project_row, tests_box,
        models_box, model_rows, workload_box, advanced_toggle, test_widgets,
        test_labels, {}, ttk, custom_models, model_vars, model_defaults, tk,
    )
    screen.render_model_rows()
    return screen


def _build_discovery(ttk, form, discovery: dict) -> None:
    box = ttk.LabelFrame(form, text="System inventory and preflight", padding=12)
    box.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
    for row, (label, value) in enumerate((
        ("System", discovery["system"]), ("Installed models", discovery["models"]),
        ("Storage", discovery["storage"]), ("Memory-fit context", discovery["memory_risk"]),
        ("llama.cpp tools", discovery["runtime"]), ("ComfyUI", discovery["comfyui"]),
    )):
        ttk.Label(box, text=f"{label}:", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, sticky="nw", padx=(0, 10), pady=2,
        )
        ttk.Label(box, text=value, wraplength=780).grid(row=row, column=1, sticky="w", pady=2)
    message = "Ready to configure a benchmark." if not discovery["issues"] else "\n".join(
        f"• {issue}" for issue in discovery["issues"]
    )
    ttk.Label(box, text=message, wraplength=900).grid(
        row=6, column=0, columnspan=2, sticky="w", pady=(8, 0),
    )


def _build_tests(ttk, parent, tests, variables, defaults):
    box = ttk.LabelFrame(parent, text="Tests", padding=12)
    box.grid(row=3, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
    box.columnconfigure(0, weight=1)
    widgets, labels = {}, {}
    for row, (name, label, _, _) in enumerate(TEST_DEFINITIONS):
        entry = next(item for item in tests if item.value == name)
        option_row = ttk.Frame(box)
        option_row.grid(row=row, column=0, sticky="ew", pady=2)
        option_row.columnconfigure(1, weight=1)
        text = label if entry.available else f"{label} (model not installed)"
        widget = ttk.Checkbutton(option_row, text=text, variable=variables[name])
        widget.grid(row=0, column=0, columnspan=2, sticky="nw")
        ttk.Button(
            box, text="Reset", width=6,
            command=lambda key=name: variables[key].set(defaults[key]),
        ).grid(row=row, column=1, sticky="e", padx=(8, 0))
        widgets[name], labels[name] = widget, widget
    return box, widgets, labels


def _build_workload(ttk, parent, cap_var, tg_vars):
    box = ttk.LabelFrame(parent, text="Workload sizes", padding=12)
    box.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
    ttk.Label(box, text="Maximum prompt-processing size").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        box, state="readonly", textvariable=cap_var,
        values=["No cap", *[str(value) for value in MAX_PROMPT_TOKEN_OPTIONS]], width=18,
    ).grid(row=0, column=1, sticky="w", padx=(10, 0))
    ttk.Label(box, text="llama-bench generation sizes").grid(row=1, column=0, sticky="w", pady=(10, 0))
    token_frame = ttk.Frame(box)
    token_frame.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
    for column, value in enumerate(TG_TOKEN_OPTIONS):
        ttk.Checkbutton(token_frame, text=str(value), variable=tg_vars[value]).grid(
            row=0, column=column, padx=(0, 8),
        )
    return box


def confirm_plan_preview(root, tk, ttk, preview: str) -> bool:
    dialog = tk.Toplevel(root)
    dialog.title("Review benchmark plan")
    dialog.geometry("760x620")
    dialog.minsize(620, 460)
    dialog.transient(root)
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(1, weight=1)
    header = ttk.Frame(dialog)
    header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 12))
    ttk.Label(header, text="Review benchmark plan", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        header, text="Confirm the resolved workload, measurement settings, and output before starting.",
    ).pack(anchor="w", pady=(4, 0))
    body = ttk.Frame(dialog)
    body.grid(row=1, column=0, sticky="nsew", padx=20)
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    text_widget = tk.Text(body, wrap="word", padx=14, pady=12, borderwidth=1, relief="solid")
    scrollbar = ttk.Scrollbar(body, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    text_widget.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    text_widget.tag_configure(
        "heading", font=("TkDefaultFont", 12, "bold"), spacing1=10, spacing3=4,
    )
    text_widget.tag_configure("label", font=("TkDefaultFont", 10, "bold"))
    text_widget.tag_configure("value", lmargin1=12, lmargin2=12, spacing3=5)
    for title, lines in plan_preview_sections(preview):
        text_widget.insert("end", f"{title}\n", "heading")
        for line in lines:
            label, separator, value = line.partition(":")
            if separator:
                text_widget.insert("end", f"{label}: ", "label")
                text_widget.insert("end", f"{value.strip()}\n", "value")
            else:
                text_widget.insert("end", f"{line}\n", "value")
    text_widget.configure(state="disabled")
    confirmed = [False]

    def finish(value: bool) -> None:
        confirmed[0] = value
        dialog.destroy()

    actions = ttk.Frame(dialog)
    actions.grid(row=2, column=0, sticky="e", padx=20, pady=18)
    ttk.Button(actions, text="Cancel", command=lambda: finish(False)).pack(side="left")
    start = ttk.Button(
        actions, text="Start Benchmark", style="Start.TButton", command=lambda: finish(True),
    )
    start.pack(side="left", padx=(10, 0))
    dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
    dialog.bind("<Escape>", lambda _event: finish(False))
    dialog.bind("<Return>", lambda _event: finish(True))
    dialog.grab_set()
    start.focus_set()
    dialog.lift()
    root.wait_window(dialog)
    return confirmed[0]
