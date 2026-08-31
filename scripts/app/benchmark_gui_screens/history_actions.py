"""History screen actions and local result browsing."""

import json
import platform
import subprocess
import threading
from pathlib import Path

from scripts.runtime import config
from scripts.app.recovery_actions import format_recovery_inspection, fork_review_report
from scripts.app.result_actions import dashboard_launcher_command, selected_result_paths
from scripts.results.acceptance_policy import evaluate_policy, load_policy
from scripts.results.recovery_inspector import inspect_recovery
from scripts.results.result_history import (
    delete_multiple_run_artifacts, discover_results, existing_run_artifacts,
    filter_results, load_result,
)
from scripts.results.vendor_diagnostic import write_vendor_diagnostic


class HistoryActions:
    def __init__(
            self, screen, *, root, tk, ttk, filedialog, messagebox,
            process_active, review_outbound_metadata, start_recovery):
        self.screen = screen
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.process_active = process_active
        self.review_outbound_metadata = review_outbound_metadata
        self.start_recovery = start_recovery
        self.entries = {"all": [], "visible": []}
        self.item_paths = {}

    def bind(self) -> None:
        self.ttk.Button(
            self.screen.filters, text="Refresh", command=self.refresh,
        ).pack(side="right")
        buttons = (
            ("Open in Dashboard", self.open_in_dashboard),
            ("Delete", self.delete_selection),
            ("Evaluate Policy", self.evaluate_selection),
            ("Export Diagnostic", self.export_diagnostic),
        )
        for index, (label, command) in enumerate(buttons):
            self.ttk.Button(self.screen.review_actions, text=label, command=command).pack(
                side="left", padx=(8, 0) if index else 0,
            )
        recovery_buttons = (
            ("Inspect Recovery", "inspect"), ("Resume", "resume"),
            ("Retry Cases", "retry"), ("Fork", "fork"),
        )
        for index, (label, action) in enumerate(recovery_buttons):
            self.ttk.Button(
                self.screen.recovery_actions, text=label,
                command=lambda value=action: self.inspect_recovery(value),
            ).pack(side="left", padx=(8, 0) if index else 0)
        self.screen.query.trace_add("write", self.apply_filters)
        self.screen.status_filter.trace_add("write", self.apply_filters)
        self.screen.engine_filter.trace_add("write", self.apply_filters)
        self.refresh()

    def selected_items(self):
        tree = self.screen.tree
        return sorted(tree.selection(), key=tree.index)

    def selected_path(self):
        return selected_result_paths(self.selected_items(), self.item_paths, exact=1)[0]

    def open_in_dashboard(self) -> None:
        try:
            paths = selected_result_paths(self.selected_items(), self.item_paths, maximum=6)
            command = dashboard_launcher_command(paths, platform.system())
            subprocess.Popen(
                command, cwd=config.SCRIPT_DIR,
                creationflags=(getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                               if platform.system() == "Windows" else 0),
            )
            suffix = "s" if len(paths) != 1 else ""
            self.screen.message.set(f"Opening {len(paths)} selected result{suffix} in the dashboard.")
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Dashboard launch failed", str(exc), parent=self.root)

    def delete_selection(self) -> None:
        if self.process_active():
            self.messagebox.showerror("Benchmark active", "Stop the active process first.", parent=self.root)
            return
        try:
            result_paths = selected_result_paths(self.selected_items(), self.item_paths)
            artifact_sets = [
                (path, existing_run_artifacts(path, config.RESULTS_DIR)) for path in result_paths
            ]
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Delete run", str(exc), parent=self.root)
            return
        artifact_sets = [(path, artifacts) for path, artifacts in artifact_sets if artifacts]
        if not artifact_sets:
            self.refresh()
            self.messagebox.showinfo("Delete run", "The selected run no longer exists.", parent=self.root)
            return
        artifact_count = sum(len(artifacts) for _, artifacts in artifact_sets)
        names = "\n".join(
            f"  • {path.name} ({len(artifacts)} artifact(s))" for path, artifacts in artifact_sets
        )
        if not self.messagebox.askyesno(
            "Delete benchmark runs",
            f"Permanently delete {len(artifact_sets)} selected run(s) and all "
            f"{artifact_count} artifact(s)?\n\n{names}\n\nThis cannot be undone. Separately "
            "exported bundles and reports are not deleted.", parent=self.root,
        ):
            return
        removed, failures = delete_multiple_run_artifacts(
            [path for path, _ in artifact_sets], config.RESULTS_DIR,
        )
        self.refresh()
        if failures:
            detail = "\n".join(f"{path.name}: {reason}" for path, reason in failures.items())
            self.messagebox.showerror(
                "Run deletion incomplete",
                f"Deleted {len(removed)} artifact(s), but some could not be removed. "
                f"The main result was retained when possible so deletion can be retried.\n\n{detail}",
                parent=self.root,
            )
            return
        self.screen.message.set(
            f"Deleted {len(artifact_sets)} run(s) and {len(removed)} artifact(s).",
        )

    def show_details(self, title, content) -> None:
        dialog = self.tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("920x620")
        dialog.transient(self.root)
        text_widget = self.tk.Text(dialog, wrap="none", font=("TkFixedFont", 10))
        y_scroll = self.ttk.Scrollbar(dialog, orient="vertical", command=text_widget.yview)
        x_scroll = self.ttk.Scrollbar(dialog, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")
        text_widget.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=(12, 0))
        y_scroll.grid(row=0, column=1, sticky="ns", pady=(12, 0))
        x_scroll.grid(row=1, column=0, sticky="ew", padx=(12, 0))
        self.ttk.Button(dialog, text="Close", command=dialog.destroy).grid(
            row=2, column=0, columnspan=2, pady=12,
        )
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(0, weight=1)

    def apply_filters(self, *_args) -> None:
        visible = filter_results(
            self.entries["all"], query=self.screen.query.get(),
            status=self.screen.status_filter.get(), engine=self.screen.engine_filter.get(),
        )
        self.entries["visible"] = visible
        tree = self.screen.tree
        tree.delete(*tree.get_children())
        self.item_paths.clear()
        for index, entry in enumerate(visible):
            item_id = tree.insert("", "end", values=(
                entry["started_at"], entry["system"], entry["status"], entry["engine"],
                entry["runtime_backend"], entry["mtp"], entry["methodology_profile"],
                entry["models_with_results"],
            ), tags=("history_even" if index % 2 == 0 else "history_odd",))
            self.item_paths[item_id] = entry["path"]
        if tree.get_children():
            tree.focus(tree.get_children()[0])
        self.screen.message.set(
            f"Showing {len(visible)} of {len(self.entries['all'])} local results. "
            "Keyboard: Shift+Up/Down extends selection; Space toggles a row.",
        )

    def refresh(self) -> None:
        entries, skipped = discover_results(config.RESULTS_DIR)
        self.entries["all"] = entries
        engines = sorted({entry["engine"] for entry in entries})
        self.screen.engine_combo.configure(values=("all", *engines))
        if self.screen.engine_filter.get() not in {"all", *engines}:
            self.screen.engine_filter.set("all")
        self.apply_filters()
        if skipped:
            self.screen.message.set(
                f"Showing {len(self.entries['visible'])} results; ignored {len(skipped)} unreadable/non-result JSON files.",
            )

    def evaluate_selection(self) -> None:
        try:
            result_path = self.selected_path()
            policy_path = self.filedialog.askopenfilename(
                title="Choose acceptance policy", filetypes=[("Acceptance policy", "*.json")],
            )
            if not policy_path:
                return
            evaluation = evaluate_policy(load_result(result_path), load_policy(Path(policy_path)))
            lines = [f"Decision: {evaluation['decision'].upper()}", ""]
            lines.extend(
                f"{item['id']}: {item['status']} (actual={item['actual']}, threshold={item['threshold']}, evidence={item['evidence']})"
                for item in evaluation["rules"]
            )
            self.show_details("Acceptance evaluation", "\n".join(lines))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Acceptance evaluation failed", str(exc), parent=self.root)

    def export_diagnostic(self) -> None:
        try:
            baseline_path, candidate_path = selected_result_paths(
                self.selected_items(), self.item_paths, exact=2,
            )
            baseline = load_result(baseline_path)
            candidate = load_result(candidate_path)
            if self.review_outbound_metadata(
                    baseline, "diagnostic baseline", allow_aliases=False) is None:
                return
            if self.review_outbound_metadata(
                    candidate, "diagnostic candidate", allow_aliases=False) is None:
                return
            destination = self.filedialog.asksaveasfilename(
                title="Export vendor diagnostic", defaultextension=".labdiag",
                filetypes=[("Local AI Bench diagnostic", "*.labdiag")],
            )
            if not destination:
                return
            write_vendor_diagnostic(baseline_path, candidate_path, Path(destination))
            self.messagebox.showinfo("Vendor diagnostic created", destination, parent=self.root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Vendor diagnostic failed", str(exc), parent=self.root)

    def inspect_recovery(self, action="inspect") -> None:
        try:
            result_path = self.selected_path()
        except ValueError as exc:
            self.messagebox.showerror("Recovery selection", str(exc), parent=self.root)
            return
        self.screen.message.set(f"Verifying recovery identity for {result_path.name}…")

        def worker():
            try:
                report, error = inspect_recovery(result_path), None
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                if action == "fork":
                    try:
                        report, error = fork_review_report(result_path), None
                    except (OSError, KeyError, ValueError, json.JSONDecodeError) as fork_exc:
                        report, error = None, str(fork_exc)
                else:
                    report, error = None, str(exc)

            def finish():
                if error or report is None:
                    self.screen.message.set("Recovery inspection failed.")
                    self.messagebox.showerror(
                        "Recovery inspection failed", error or "No recovery report was produced.",
                        parent=self.root,
                    )
                    return
                self.screen.message.set(
                    f"Recovery decision for {result_path.name}: {report['action']}",
                )
                if action == "inspect":
                    self.show_details("Recovery inspection", format_recovery_inspection(report))
                    return
                if action in {"resume", "retry"} and not report["can_resume"]:
                    self.show_details("Fork required", format_recovery_inspection(report))
                    return
                selected = self.choose_retry_cases(report.get("retryable_cases", [])) \
                    if action == "retry" else None
                if action == "retry" and not selected:
                    return
                self.start_recovery(action, result_path, report, selected)

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def choose_retry_cases(self, candidates):
        by_stage = {}
        for candidate in candidates:
            by_stage.setdefault(candidate["stage"], []).append(candidate)
        if not by_stage:
            self.messagebox.showinfo("Selected retry", "No cases are retry-eligible.", parent=self.root)
            return []
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Select cases to retry")
        dialog.transient(self.root)
        dialog.grab_set()
        shell = self.ttk.Frame(dialog, padding=16)
        shell.pack(fill="both", expand=True)
        self.ttk.Label(
            shell, text="Retry only the chosen cases. Other incomplete evidence remains unchanged.",
            wraplength=520,
        ).pack(anchor="w", pady=(0, 8))
        stage_var = self.tk.StringVar(value=next(iter(by_stage)))
        stage_picker = self.ttk.Combobox(
            shell, textvariable=stage_var, values=list(by_stage), state="readonly",
        )
        stage_picker.pack(fill="x", pady=(0, 8))
        case_list = self.tk.Listbox(shell, selectmode="extended", width=72, height=12)
        case_list.pack(fill="both", expand=True)

        def refresh_cases(_event=None):
            case_list.delete(0, "end")
            for candidate in by_stage[stage_var.get()]:
                case_list.insert("end", f"{candidate['label']} — {candidate['state']}")

        selected = []

        def accept():
            selected.extend(by_stage[stage_var.get()][index] for index in case_list.curselection())
            if not selected:
                self.messagebox.showerror("Selected retry", "Select at least one case.", parent=dialog)
                return
            dialog.destroy()

        stage_picker.bind("<<ComboboxSelected>>", refresh_cases)
        refresh_cases()
        buttons = self.ttk.Frame(shell)
        buttons.pack(fill="x", pady=(10, 0))
        self.ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        self.ttk.Button(buttons, text="Retry Selected", command=accept).pack(
            side="right", padx=(0, 8),
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.root.wait_window(dialog)
        return selected
