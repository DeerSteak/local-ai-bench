"""Run Log screen actions for result and support artifacts."""

import json
import platform
import subprocess
from pathlib import Path

from scripts.runtime import config
from scripts.app.benchmark_gui_process import open_path_command
from scripts.app.result_actions import result_paths_for_log, run_log_path
from scripts.results.acceptance_policy import load_policy
from scripts.results.decision_report import (
    load_result, report_output_paths, write_html_report, write_pdf_report,
)
from scripts.results.outbound_metadata import outbound_metadata_preview, prepare_outbound_result
from scripts.results.result_bundle import export_result_bundle, import_result_bundle, verify_result_bundle
from scripts.results.support_bundle import export_support_bundle, preview_support_bundle


class RunLogActions:
    def __init__(
            self, screen, *, root, tk, ttk, filedialog, messagebox, option_vars,
            active_project, active_result_paths):
        self.screen = screen
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.option_vars = option_vars
        self.active_project = active_project
        self.active_result_paths = active_result_paths

    def bind(self) -> None:
        actions = self.screen.result_actions
        buttons = (
            ("Open Results Folder", self.open_results_folder),
            ("Export Log", self.export_log),
            ("Export Bundle", self.export_bundle),
            ("Import / Verify", self.import_bundle),
            ("Create Report", self.create_report),
            ("Support Bundle", self.export_support),
        )
        for index, (label, command) in enumerate(buttons):
            self.ttk.Button(actions, text=label, command=command).pack(
                side="left", padx=(8, 0) if index else 0,
            )

    def current_log(self) -> str:
        return self.screen.current_log()

    def export_log(self) -> None:
        log = self.current_log()
        if not log:
            self.messagebox.showinfo("Export Log", "The Run Log is empty.", parent=self.root)
            return
        known_results = result_paths_for_log(log, self.active_result_paths()[:1])
        suggested = run_log_path(known_results[0]) if known_results else config.RESULTS_DIR / "run_log.txt"
        destination = self.filedialog.asksaveasfilename(
            title="Export Run Log", initialdir=str(suggested.parent),
            initialfile=suggested.name, defaultextension=".txt",
            filetypes=[("Text log", "*.txt"), ("All files", "*")],
        )
        if not destination:
            return
        try:
            Path(destination).write_text(log, encoding="utf-8")
            self.messagebox.showinfo("Log exported", f"Run Log saved to:\n{destination}", parent=self.root)
        except OSError as exc:
            self.messagebox.showerror("Log export failed", str(exc), parent=self.root)

    def open_results_folder(self) -> None:
        output = self.option_vars["out"].get().strip()
        folder = Path(output).expanduser().resolve().parent if output else config.RESULTS_DIR
        subprocess.Popen(open_path_command(folder, platform.system()))

    def review_outbound_metadata(self, result, purpose, *, allow_aliases=True):
        decision: dict[str, dict | None] = {"value": None}
        dialog = self.tk.Toplevel(self.root)
        dialog.title(f"Review metadata for {purpose}")
        dialog.geometry("760x600")
        dialog.transient(self.root)
        dialog.grab_set()
        self.ttk.Label(
            dialog,
            text="Review every identity field before it leaves this machine. Optional aliases replace exported names only; the source result is unchanged.",
            wraplength=710,
        ).pack(anchor="w", padx=16, pady=(16, 8))
        preview_frame = self.ttk.Frame(dialog)
        preview_frame.pack(fill="both", expand=True, padx=16)
        text_widget = self.tk.Text(preview_frame, wrap="none", height=20)
        scroll = self.ttk.Scrollbar(preview_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scroll.set)
        text_widget.insert("1.0", "\n".join(
            f"{label}: {value}" for label, value in outbound_metadata_preview(result)
        ))
        text_widget.configure(state="disabled")
        text_widget.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        aliases = self.ttk.LabelFrame(dialog, text="Optional private aliases", padding=10)
        if allow_aliases:
            aliases.pack(fill="x", padx=16, pady=(10, 0))
        system_alias = self.tk.StringVar()
        hardware_alias = self.tk.StringVar()
        self.ttk.Label(aliases, text="System name").grid(row=0, column=0, sticky="w")
        self.ttk.Entry(aliases, textvariable=system_alias).grid(
            row=0, column=1, sticky="ew", padx=(10, 0),
        )
        self.ttk.Label(aliases, text="Hardware name").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.ttk.Entry(aliases, textvariable=hardware_alias).grid(
            row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 0),
        )
        aliases.columnconfigure(1, weight=1)
        actions = self.ttk.Frame(dialog)
        actions.pack(fill="x", padx=16, pady=16)

        def approve():
            decision["value"] = {
                "system_alias": system_alias.get().strip() or None,
                "hardware_alias": hardware_alias.get().strip() or None,
            }
            dialog.destroy()

        self.ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right")
        self.ttk.Button(actions, text="Approve Export", command=approve).pack(
            side="right", padx=(0, 8),
        )
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.root.wait_window(dialog)
        return decision["value"]

    def export_bundle(self) -> None:
        result = self.filedialog.askopenfilename(
            title="Choose result JSON", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result:
            return
        try:
            source = load_result(Path(result))
            aliases = self.review_outbound_metadata(source, "result bundle")
            if aliases is None:
                return
        except (OSError, ValueError, KeyError) as exc:
            self.messagebox.showerror("Bundle export failed", str(exc), parent=self.root)
            return
        bundle = self.filedialog.asksaveasfilename(
            title="Export verified result bundle", defaultextension=".labresult",
            filetypes=[("Local AI Bench result", "*.labresult")],
        )
        if not bundle:
            return
        try:
            export_result_bundle(Path(result), Path(bundle), **aliases)
            self.messagebox.showinfo("Bundle exported", f"Verified bundle saved to:\n{bundle}", parent=self.root)
        except (OSError, ValueError, KeyError) as exc:
            self.messagebox.showerror("Bundle export failed", str(exc), parent=self.root)

    def import_bundle(self) -> None:
        bundle = self.filedialog.askopenfilename(
            title="Import and verify result bundle",
            filetypes=[("Local AI Bench result", "*.labresult")],
        )
        if not bundle:
            return
        destination = self.filedialog.asksaveasfilename(
            title="Save verified result JSON", initialdir=config.RESULTS_DIR,
            defaultextension=".json", filetypes=[("Benchmark result", "*.json")],
        )
        if not destination:
            return
        try:
            verify_result_bundle(Path(bundle))
            import_result_bundle(Path(bundle), Path(destination), Path(destination).with_suffix(""))
            self.messagebox.showinfo(
                "Bundle verified and imported", f"Verified result saved to:\n{destination}", parent=self.root,
            )
        except (OSError, ValueError, KeyError) as exc:
            self.messagebox.showerror("Bundle verification failed", str(exc), parent=self.root)

    def create_report(self) -> None:
        result_path = self.filedialog.askopenfilename(
            title="Choose result JSON", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result_path:
            return
        try:
            source_result = load_result(Path(result_path))
            aliases = self.review_outbound_metadata(source_result, "decision report")
            if aliases is None:
                return
        except (OSError, ValueError, KeyError) as exc:
            self.messagebox.showerror("Report creation failed", str(exc), parent=self.root)
            return
        destination = self.filedialog.asksaveasfilename(
            title="Save decision report", initialdir=config.RESULTS_DIR,
            defaultextension=".html", filetypes=[("Decision report", "*.html")],
        )
        if not destination:
            return
        try:
            html_path, pdf_path = report_output_paths(Path(destination))
            result = prepare_outbound_result(source_result, **aliases)
            policy = (self.active_project["value"] or {}).get("acceptance_policy")
            if policy is None and self.messagebox.askyesno(
                    "Acceptance policy", "Apply an acceptance policy to this report?", parent=self.root):
                policy_path = self.filedialog.askopenfilename(
                    title="Choose acceptance policy", filetypes=[("Acceptance policy", "*.json")],
                )
                if not policy_path:
                    return
                policy = load_policy(Path(policy_path))
            write_html_report(result, html_path, policy)
            write_pdf_report(result, pdf_path, policy)
            self.messagebox.showinfo(
                "Decision report created", f"HTML and PDF reports saved to:\n{html_path.parent}",
                parent=self.root,
            )
        except (OSError, ValueError, KeyError) as exc:
            self.messagebox.showerror("Report creation failed", str(exc), parent=self.root)

    def confirm_support_preview(self, preview) -> bool:
        accepted = {"value": False}
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Review redacted support bundle")
        dialog.geometry("720x520")
        dialog.transient(self.root)
        dialog.grab_set()
        self.ttk.Label(
            dialog, text="Review every file and field before export. Raw results and logs are not included.",
            wraplength=680,
        ).pack(anchor="w", padx=16, pady=(16, 8))
        text_widget = self.tk.Text(dialog, wrap="none", height=22)
        scrollbar = self.ttk.Scrollbar(dialog, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.pack(side="left", fill="both", expand=True, padx=(16, 0), pady=(0, 16))
        scrollbar.pack(side="left", fill="y", pady=(0, 16))
        details = "Files:\n" + "\n".join(f"  {name}" for name in preview["files"])
        details += "\n\nFields:\n" + "\n".join(f"  {field}" for field in preview["fields"])
        text_widget.insert("1.0", details)
        text_widget.configure(state="disabled")
        actions = self.ttk.Frame(dialog, padding=(8, 16))
        actions.pack(side="right", fill="y")

        def accept():
            accepted["value"] = True
            dialog.destroy()

        self.ttk.Button(actions, text="Export", command=accept).pack(fill="x", pady=(0, 8))
        self.ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(fill="x")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        self.root.wait_window(dialog)
        return accepted["value"]

    def export_support(self) -> None:
        result = self.filedialog.askopenfilename(
            title="Choose result for support bundle", initialdir=config.RESULTS_DIR,
            filetypes=[("Benchmark result", "*.json")],
        )
        if not result:
            return
        try:
            preview = preview_support_bundle(Path(result))
            if not self.confirm_support_preview(preview):
                return
            destination = self.filedialog.asksaveasfilename(
                title="Export redacted support bundle", defaultextension=".labsupport",
                filetypes=[("Local AI Bench support", "*.labsupport")],
            )
            if destination:
                export_support_bundle(Path(result), Path(destination))
                self.messagebox.showinfo("Support bundle exported", destination, parent=self.root)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.messagebox.showerror("Support bundle failed", str(exc), parent=self.root)
