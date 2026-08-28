"""Configuration screen live state and imported-model coordination."""

from scripts.runtime import config
from scripts.app.benchmark_frontend import (
    build_model_entries, build_test_entries, merge_model_inventories,
    parse_engine_selection,
)
from scripts.app.benchmark_gui_support import (
    CUSTOM_PRESET, GPU_SPLIT_MODE_LABELS, MTP_MODE_LABELS,
    preset_after_control_change, preset_control_values,
    reconcile_imported_model_state,
)
from scripts.app.model_import_dialog import show_model_import_dialog
from scripts.app.tk_utils import mousewheel_scroll_units
from scripts.runtime.engines import get_engine
from scripts.setup.model_inventory import build_model_inventory


class ConfigurationStateController:
    def __init__(
            self, screen, *, root, tk, ttk, messagebox, advanced_var, engine_var,
            test_vars, model_vars, cap_var, tg_vars, option_vars, preset_var,
            available_engines, custom_tests, custom_models, defaults_for_display,
            applying_configuration, engine_inventories, inventory, model_owners,
            custom_model_defaults, set_selected_engines, apply_engine_availability,
            execution_box, paths_box):
        self.screen = screen
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.advanced_var = advanced_var
        self.engine_var = engine_var
        self.test_vars = test_vars
        self.model_vars = model_vars
        self.cap_var = cap_var
        self.tg_vars = tg_vars
        self.option_vars = option_vars
        self.preset_var = preset_var
        self.available_engines = available_engines
        self.custom_tests = custom_tests
        self.custom_models = custom_models
        self.defaults_for_display = defaults_for_display
        self.applying_configuration = applying_configuration
        self.engine_inventories = engine_inventories
        self.inventory = inventory
        self.model_owners = model_owners
        self.custom_model_defaults = custom_model_defaults
        self.set_selected_engines = set_selected_engines
        self.apply_engine_availability = apply_engine_availability
        self.execution_box = execution_box
        self.paths_box = paths_box
        self.last_signature = [self.control_signature()]

    def bind(self) -> None:
        self.preset_var.trace_add("write", self.select_preset)
        for variable in (
                *self.test_vars.values(), *self.model_vars.values(), self.engine_var,
                self.cap_var, *self.tg_vars.values(), *self.option_vars.values()):
            variable.trace_add("write", self.mark_custom)
        self.advanced_var.trace_add("write", lambda *_: self.update_advanced())
        for entry in self.custom_tests:
            if not entry.available:
                self.screen.test_widgets[entry.value].configure(state="disabled")
        if self.preset_var.get() != CUSTOM_PRESET:
            self.apply_named_preset(self.preset_var.get())
        self.root.bind_all("<MouseWheel>", self.scroll_form)
        self.root.bind_all("<Button-4>", self.scroll_form)
        self.root.bind_all("<Button-5>", self.scroll_form)

    def apply_control_values(self, values: dict) -> None:
        for name, variable in self.test_vars.items():
            variable.set(values["tests"].get(name, False))
        if "models" in values:
            for name, variable in self.model_vars.items():
                variable.set(values["models"].get(name, False))
        if values.get("engine"):
            self.set_selected_engines(parse_engine_selection(values["engine"]))
        self.cap_var.set(values["max_prompt_tokens"])
        for value, variable in self.tg_vars.items():
            variable.set(value in values["tg_tokens"])
        for key, value in values["options"].items():
            display_value = (
                GPU_SPLIT_MODE_LABELS[value] if key == "gpu_split_mode"
                else MTP_MODE_LABELS[value] if key == "mtp" else value
            )
            self.option_vars[key].set(display_value)

    def control_signature(self) -> tuple:
        return (
            tuple(variable.get() for variable in self.test_vars.values()),
            tuple(variable.get() for variable in self.model_vars.values()),
            self.engine_var.get(), self.cap_var.get(),
            tuple(variable.get() for variable in self.tg_vars.values()),
            tuple(variable.get() for variable in self.option_vars.values()),
        )

    def apply_named_preset(self, name: str) -> None:
        if name == CUSTOM_PRESET:
            return
        available = {entry.value for entry in self.custom_tests if entry.available}
        self.applying_configuration[0] = True
        try:
            values = preset_control_values(name, available, self.defaults_for_display)
            self.apply_control_values(values)
        finally:
            self.applying_configuration[0] = False

    def mark_custom(self, *_args) -> None:
        signature = self.control_signature()
        changed = signature != self.last_signature[0]
        self.last_signature[0] = signature
        if not changed:
            return
        updated = preset_after_control_change(
            self.preset_var.get(), self.applying_configuration[0],
        )
        if updated != self.preset_var.get():
            self.preset_var.set(updated)

    def select_preset(self, *_args) -> None:
        self.apply_named_preset(self.preset_var.get())

    def update_advanced(self) -> None:
        for box in (self.execution_box, self.paths_box):
            box.grid() if self.advanced_var.get() else box.grid_remove()

    def refresh_imported_models(self, selected_tag: str) -> None:
        previous_values = set(self.model_vars)
        previous_selected = {
            name for name, variable in self.model_vars.items() if variable.get()
        }
        refreshed = {
            name: build_model_inventory(get_engine(name), config.COMFYUI_MODELS_DIR)
            for name in self.available_engines
        }
        merged, owners = merge_model_inventories(refreshed)
        self.engine_inventories.clear()
        self.engine_inventories.update(refreshed)
        self.inventory.clear()
        self.inventory.update(merged)
        self.model_owners.clear()
        self.model_owners.update(owners)
        refreshed_tests = {entry.value: entry for entry in build_test_entries(self.inventory)}
        for entry in self.custom_tests:
            entry.available = refreshed_tests[entry.value].available
            self.screen.test_widgets[entry.value].configure(
                state="normal" if entry.available else "disabled",
            )
            self.screen.test_labels[entry.value].configure(
                text=entry.label if entry.available else f"{entry.label} (model not installed)",
            )
        rebuilt = build_model_entries(
            self.inventory, [entry.value for entry in self.custom_tests if entry.available],
        )
        self.custom_models[:] = rebuilt
        selected, dropped, added, defaults = reconcile_imported_model_state(
            previous_values, previous_selected, self.custom_model_defaults, rebuilt, selected_tag,
        )
        for value in dropped:
            self.model_vars.pop(value)
        self.custom_model_defaults.clear()
        self.custom_model_defaults.update(defaults)
        for entry in rebuilt:
            if entry.value in added:
                self.model_vars[entry.value] = self.tk.BooleanVar(value=False)
                self.model_vars[entry.value].trace_add("write", self.mark_custom)
            self.model_vars[entry.value].set(entry.value in selected)
        self.screen.render_model_rows()
        self.apply_engine_availability()

    def open_model_import_dialog(self) -> None:
        show_model_import_dialog(
            root=self.root, tk=self.tk, ttk=self.ttk, messagebox=self.messagebox,
            available_engines=self.available_engines, engine_factory=get_engine,
            on_imported=self.refresh_imported_models,
        )

    def scroll_form(self, event):
        try:
            widget = self.root.winfo_containing(
                self.root.winfo_pointerx(), self.root.winfo_pointery(),
            )
        except (KeyError, self.tk.TclError):
            return None
        current = widget
        while current is not None and current not in {self.screen.canvas, self.screen.form}:
            current = getattr(current, "master", None)
        if current is None:
            return None
        units = mousewheel_scroll_units(
            delta=getattr(event, "delta", 0), button=getattr(event, "num", 0),
            platform_name=self.root.tk.call("tk", "windowingsystem"),
        )
        if units:
            self.screen.canvas.yview_scroll(units, "units")
        return "break"
