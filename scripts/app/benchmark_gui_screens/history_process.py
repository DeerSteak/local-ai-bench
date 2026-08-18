"""History recovery, retry, and fork process preparation."""

from pathlib import Path

from scripts.runtime import config
from scripts.app.recovery_actions import (
    fork_executor_command, format_recovery_inspection, recovery_executor_command,
    recovery_progress_entries, retry_executor_command,
)
from scripts.results.run_plan import load_run_plan
from scripts.stage_registry import JOURNAL_STAGES


class HistoryProcessActions:
    def __init__(self, *, root, filedialog, messagebox, process_active, launch):
        self.root = root
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.process_active = process_active
        self.launch = launch

    def start(self, action, result_path, report, selected=None) -> None:
        if self.process_active():
            self.messagebox.showerror(
                "Benchmark active", "Stop the active process first.", parent=self.root,
            )
            return
        if action == "resume":
            self.resume(result_path, report)
        elif action == "retry":
            self.retry(result_path, selected or [])
        else:
            self.fork(result_path, report)

    def resume(self, result_path, report) -> None:
        plan = load_run_plan(result_path)
        unsupported = [stage for stage in plan.stage_order if stage not in JOURNAL_STAGES]
        if unsupported:
            self.messagebox.showerror(
                "Recovery unavailable",
                "This saved plan contains stages without durable recovery: "
                + ", ".join(unsupported), parent=self.root,
            )
            return
        if not self.messagebox.askyesno(
            "Resume stopped benchmark",
            f"{format_recovery_inspection(report)}\n\n"
            "Resume the remaining journal-owned work in this result?", parent=self.root,
        ):
            return
        self.launch(
            recovery_executor_command(result_path), "recovery",
            [Path(result_path).resolve()],
            "Recovery is running. Completed evidence is preserved.",
            plan.stage_order, recovery_progress_entries(plan), [plan.engine_name],
            "Recovery could not start",
        )

    def fork(self, source_path, report) -> None:
        plan = load_run_plan(source_path)
        destination = self.filedialog.asksaveasfilename(
            title="Save forked benchmark", defaultextension=".json",
            initialdir=str(config.RESULTS_DIR), initialfile=f"{source_path.stem}_fork.json",
            filetypes=[("JSON results", "*.json")],
        )
        if not destination:
            return
        output_path = Path(destination).resolve()
        if not self.messagebox.askyesno(
            "Fork benchmark plan",
            f"{format_recovery_inspection(report)}\n\n"
            "Run this saved plan from the beginning as a new result? "
            "The source result will not be changed.", parent=self.root,
        ):
            return
        self.launch(
            fork_executor_command(source_path, output_path), "fork", [output_path],
            "Forked run is active. The source evidence remains unchanged.",
            plan.stage_order, recovery_progress_entries(plan), [plan.engine_name],
            "Fork could not start",
        )

    def retry(self, result_path, selected) -> None:
        if not selected:
            return
        if not self.messagebox.askyesno(
            "Retry selected cases",
            f"Retry {len(selected)} selected case(s)? Completed and unselected evidence will not rerun.",
            parent=self.root,
        ):
            return
        plan = load_run_plan(result_path)
        models = {candidate["model"] for candidate in selected}
        self.launch(
            retry_executor_command(
                result_path, [candidate["case_id"] for candidate in selected],
            ),
            "retry", [Path(result_path).resolve()],
            "Selected retry is running. Unselected evidence remains unchanged.",
            [selected[0]["stage"]], recovery_progress_entries(plan, models),
            [plan.engine_name], "Selected retry could not start",
        )
