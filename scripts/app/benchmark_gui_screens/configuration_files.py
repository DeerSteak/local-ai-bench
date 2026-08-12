"""Configuration screen preset, project, and run-plan file actions."""

import json
from pathlib import Path

from scripts.runtime import config
from scripts.app.benchmark_frontend import frontend_state_from_run_plan
from scripts.app.benchmark_presets import (
    build_portable_preset, compare_portable_presets, load_portable_preset,
    save_portable_preset,
)
from scripts.app.benchmark_project import (
    PROJECT_WORKFLOWS, build_project, load_project, project_frontend_state, save_project,
)
from scripts.results.acceptance_policy import load_policy
from scripts.results.run_plan import load_run_plan


class ConfigurationFileActions:
    def __init__(
            self, screen, *, root, tk, ttk, filedialog, simpledialog, messagebox,
            active_project, project_status, current_state, apply_state, collect_options):
        self.screen = screen
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.simpledialog = simpledialog
        self.messagebox = messagebox
        self.active_project = active_project
        self.project_status = project_status
        self.current_state = current_state
        self.apply_state = apply_state
        self.collect_options = collect_options

    def bind(self) -> None:
        preset_buttons = (
            ("Export", self.export_preset), ("Import", self.import_preset),
            ("Compare", self.compare_preset), ("Import CLI Plan", self.import_run_plan),
        )
        for label, command in preset_buttons:
            self.ttk.Button(self.screen.preset_row, text=label, command=command).pack(
                side="left", padx=(8, 0),
            )
        self.ttk.Button(
            self.screen.project_row, text="New Project", command=self.save_project,
        ).pack(side="left")
        self.ttk.Button(
            self.screen.project_row, text="Open Project", command=self.open_project,
        ).pack(side="left", padx=(8, 0))

    def export_preset(self, preset=None) -> None:
        if preset is None:
            name = self.simpledialog.askstring(
                "Preset name", "Name this portable preset:", parent=self.root,
            )
            if not name:
                return
            preset = build_portable_preset(name, self.current_state())
        path = self.filedialog.asksaveasfilename(
            title="Export benchmark preset", defaultextension=".json",
            filetypes=[("Benchmark preset", "*.json")],
        )
        if path:
            save_portable_preset(Path(path), preset)

    def apply_portable_preset(self, portable) -> None:
        configuration = portable["configuration"]
        self.apply_state({
            "tests": configuration["tests"],
            "models": configuration["models"],
            "max_prompt_tokens": configuration["max_prompt_tokens"],
            "tg_tokens": configuration["tg_tokens"],
            "gui_options": configuration["options"],
        })

    def import_preset(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Import benchmark preset", filetypes=[("Benchmark preset", "*.json")],
        )
        if not path:
            return
        try:
            self.apply_portable_preset(load_portable_preset(Path(path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Preset import failed", str(exc), parent=self.root)

    def compare_preset(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Compare with preset", filetypes=[("Benchmark preset", "*.json")],
        )
        if not path:
            return
        try:
            saved = load_portable_preset(Path(path))
            current = build_portable_preset("Current screen", self.current_state())
            differences = compare_portable_presets(current, saved)
            detail = ", ".join(differences) if differences else "No configuration differences."
            self.messagebox.showinfo("Preset comparison", detail, parent=self.root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Preset comparison failed", str(exc), parent=self.root)

    def import_run_plan(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Import CLI run plan or results", filetypes=[("Benchmark JSON", "*.json")],
        )
        if not path:
            return
        try:
            state = frontend_state_from_run_plan(
                load_run_plan(Path(path)), self.collect_options(),
            )
            self.apply_state(state)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Run-plan import failed", str(exc), parent=self.root)

    def choose_project_workflow(self):
        selected: dict[str, str | None] = {"value": None}
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Project workflow")
        dialog.transient(self.root)
        dialog.grab_set()
        workflow_var = self.tk.StringVar(value=next(iter(PROJECT_WORKFLOWS)))
        self.ttk.Label(dialog, text="What decision will this project support?").pack(
            anchor="w", padx=18, pady=(18, 8),
        )
        combo = self.ttk.Combobox(
            dialog, state="readonly", width=30, values=list(PROJECT_WORKFLOWS.values()),
        )
        combo.current(0)
        combo.pack(fill="x", padx=18)

        def accept():
            workflow_var.set(next(
                key for key, label in PROJECT_WORKFLOWS.items() if label == combo.get()
            ))
            selected["value"] = workflow_var.get()
            dialog.destroy()

        actions = self.ttk.Frame(dialog)
        actions.pack(fill="x", padx=18, pady=18)
        self.ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right")
        self.ttk.Button(actions, text="Continue", command=accept).pack(
            side="right", padx=(0, 8),
        )
        self.root.wait_window(dialog)
        return selected["value"]

    def save_project(self) -> None:
        name = self.simpledialog.askstring("New project", "Project name:", parent=self.root)
        if not name:
            return
        workflow = self.choose_project_workflow()
        if not workflow:
            return
        baseline = None
        if self.messagebox.askyesno("Baseline", "Attach an existing baseline result?", parent=self.root):
            baseline = self.filedialog.askopenfilename(
                title="Choose baseline result", initialdir=config.RESULTS_DIR,
                filetypes=[("Benchmark result", "*.json")],
            )
            if not baseline:
                return
        acceptance = None
        if self.messagebox.askyesno(
                "Acceptance policy", "Attach an acceptance policy?", parent=self.root):
            policy_path = self.filedialog.askopenfilename(
                title="Choose acceptance policy", filetypes=[("Acceptance policy", "*.json")],
            )
            if not policy_path:
                return
            try:
                acceptance = load_policy(Path(policy_path))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.messagebox.showerror("Acceptance policy failed", str(exc), parent=self.root)
                return
        destination = self.filedialog.asksaveasfilename(
            title="Save benchmark project", defaultextension=".labproject",
            filetypes=[("Local AI Bench project", "*.labproject")],
        )
        if not destination:
            return
        try:
            project = build_project(
                name, workflow, self.current_state(), baseline_result=baseline,
                acceptance_policy=acceptance,
            )
            save_project(Path(destination), project)
            self.active_project["value"] = project
            self.project_status.set(f"Project: {project['name']} ({PROJECT_WORKFLOWS[workflow]})")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Project creation failed", str(exc), parent=self.root)

    def open_project(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Open benchmark project", filetypes=[("Local AI Bench project", "*.labproject")],
        )
        if not path:
            return
        try:
            project = load_project(Path(path))
            self.apply_state(project_frontend_state(project, self.collect_options()))
            self.active_project["value"] = project
            self.project_status.set(
                f"Project: {project['name']} ({PROJECT_WORKFLOWS[project['workflow']]})",
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Project open failed", str(exc), parent=self.root)
