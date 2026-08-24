"""Benchmark progress window and live stage/model presentation."""

import time
from dataclasses import dataclass, field
from typing import Any

from scripts.app.benchmark_frontend import LLM_BACKED_TESTS, TEST_STAGE_LABELS, engine_incompatible_tests
from scripts.app.tk_utils import mousewheel_scroll_units
from scripts.runtime import config
from scripts.runtime.mtp import mtp_tests, native_mtp_config
from scripts.stage_registry import STAGE_ORDER
from scripts.workloads.models import LLM_MODELS
from scripts.workloads.model_variants import expanded_variant_catalog


def progress_engine_base(engine: str) -> str:
    return engine.partition(" · MTP ")[0]


def progress_engine_mtp_enabled(engine: str) -> bool:
    return engine.endswith(" · MTP on")


def progress_entries_for_engine(entries, engine: str,
                                model_owners: dict[str, set[str]]) -> list:
    """Selected entries compatible with one engine; images remain engine-independent."""
    base_engine = progress_engine_base(engine)
    mtp_enabled = progress_engine_mtp_enabled(engine)
    catalog = {model["tag"]: model for model in expanded_variant_catalog(LLM_MODELS)}
    return [
        entry for entry in entries if entry.checked and (
            entry.kind == "image" or entry.value not in model_owners
            or base_engine in model_owners[entry.value]
        )
        and (not mtp_enabled or native_mtp_config(catalog.get(entry.value, {}), base_engine))
    ]


@dataclass
class ProgressScreen:
    root: Any
    tk: Any
    ttk: Any
    metrics_updater: Any
    summary_builder: Any
    engine_resolver: Any
    model_identity: Any
    window: Any = None
    stage_vars: dict = field(default_factory=dict)
    model_vars: dict = field(default_factory=dict)
    summary_vars: dict = field(default_factory=dict)
    resource_vars: dict = field(default_factory=dict)
    engines: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    started_at: float | None = None
    remaining_var: Any = None

    def show(self, tests, entries, engines: list[str],
             model_owners: dict[str, set[str]] | None = None,
             *, show_vram: bool) -> None:
        if self.window is not None and self.window.winfo_exists():
            self.window.destroy()
        self.window = self.tk.Toplevel(self.root)
        self.window.title(f"Local AI Bench v{config.VERSION} Progress")
        self.window.geometry("460x640")
        self.window.minsize(380, 300)
        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.window.withdraw)
        shell = self.ttk.Frame(self.window, padding=18)
        shell.pack(fill="both", expand=True)
        self.ttk.Label(shell, text="Benchmark progress", style="Title.TLabel").pack(anchor="w")
        self.ttk.Label(
            shell, text="Live workload/model status, quality, resources, and remaining time.",
            wraplength=390,
        ).pack(anchor="w", pady=(2, 12))
        model_owners = {} if model_owners is None else model_owners
        total = 0
        selected_by_engine = {
            engine: progress_entries_for_engine(entries, engine, model_owners)
            for engine in engines
        }
        for engine_index, engine_name in enumerate(engines):
            base_engine = progress_engine_base(engine_name)
            pass_tests = mtp_tests(tests, progress_engine_mtp_enabled(engine_name))
            skipped = set(engine_incompatible_tests(pass_tests, base_engine))
            for stage in (key for key in STAGE_ORDER if key in pass_tests):
                if (stage == "img" and engine_index > 0) or stage in skipped:
                    continue
                stage_entries = self._stage_entries(stage, selected_by_engine[engine_name])
                if self._model_backed_stage(stage) and not stage_entries:
                    continue
                total += len(stage_entries)
        self.metrics = {
            "total_models": total, "finished_models": set(), "usable_models": set(),
            "retries": 0, "valid": 0, "invalid": 0, "last_completion_elapsed": None,
        }
        self.started_at = time.monotonic()
        self._build_summary(shell)
        self._build_resources(shell, show_vram)
        status_list = self._build_status_list(shell)
        self.engines[:] = engines
        self.stage_vars.clear()
        self.model_vars.clear()
        for engine_index, engine_name in enumerate(engines):
            base_engine = progress_engine_base(engine_name)
            pass_tests = mtp_tests(tests, progress_engine_mtp_enabled(engine_name))
            skipped = set(engine_incompatible_tests(pass_tests, base_engine))
            for stage in (key for key in STAGE_ORDER if key in pass_tests):
                if (stage == "img" and engine_index > 0) or stage in skipped:
                    continue
                if self._model_backed_stage(stage) and not self._stage_entries(
                        stage, selected_by_engine[engine_name]):
                    continue
                self._add_stage(
                    status_list, engine_name, stage, selected_by_engine[engine_name],
                    len(engines) > 1,
                )
        self.window.lift()

    def update(self, event: dict) -> None:
        self.metrics = self.metrics_updater(self.metrics, event)
        for label, value in self.summary_builder(self.metrics).items():
            self.summary_vars[label].set(value)
        if event["kind"] == "measurement":
            return
        engine = self.engine_resolver(event, self.engines)
        if engine is None:
            return
        variable = (
            self.model_vars.get((engine, event["stage"], self.model_identity(event)))
            if event["kind"] == "model" else self.stage_vars.get((engine, event["stage"]))
        )
        if variable is None:
            return
        variable.set({
            "running": "▶ Running", "complete": "✓ Complete", "skipped": "— Skipped",
            "failed": "✕ Failed", "interrupted": "■ Interrupted",
        }[event["status"]])
        if event["kind"] == "stage" and event["status"] in {"complete", "failed", "interrupted"}:
            for (row_engine, stage, _), model_var in self.model_vars.items():
                if row_engine == engine and stage == event["stage"] and model_var.get() in {"○ Queued", "▶ Running"}:
                    model_var.set("■ Interrupted" if event["status"] == "interrupted" else "— Not run")

    def set_resources(self, rows: dict[str, str], remaining: str) -> None:
        for label, value in rows.items():
            if label in self.resource_vars:
                self.resource_vars[label].set(value)
        self.remaining_var.set(f"Remaining time: {remaining}")

    def finish_pending(self, exit_code: int) -> None:
        for variable in self.stage_vars.values():
            if variable.get() in {"○ Queued", "▶ Running"}:
                variable.set("— Not run" if exit_code else "✓ Complete")
        for variable in self.model_vars.values():
            if variable.get() in {"○ Queued", "▶ Running"}:
                variable.set("— Not run")

    def _build_summary(self, shell) -> None:
        box = self.ttk.LabelFrame(shell, text="Run summary", padding=(10, 6))
        box.pack(fill="x", pady=(0, 8))
        self.summary_vars.clear()
        for row, (label, value) in enumerate(self.summary_builder(self.metrics).items()):
            self.ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=1)
            variable = self.tk.StringVar(value=value)
            self.summary_vars[label] = variable
            self.ttk.Label(box, textvariable=variable).grid(row=row, column=1, sticky="w", pady=1)

    def _build_resources(self, shell, show_vram: bool) -> None:
        box = self.ttk.LabelFrame(shell, text="Resources", padding=(10, 6))
        box.pack(fill="x", pady=(0, 8))
        self.resource_vars.clear()
        labels = ["CPU", "Process RAM", "System RAM", "GPU", *(["VRAM"] if show_vram else [])]
        for row, label in enumerate(labels):
            self.ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14), pady=1)
            variable = self.tk.StringVar(value="Starting…")
            self.resource_vars[label] = variable
            self.ttk.Label(box, textvariable=variable).grid(row=row, column=1, sticky="w", pady=1)
        self.remaining_var = self.tk.StringVar(value="Remaining time: calibrating")
        self.ttk.Label(shell, textvariable=self.remaining_var).pack(anchor="w", pady=(0, 8))

    def _build_status_list(self, shell):
        holder = self.ttk.Frame(shell)
        holder.pack(fill="both", expand=True)
        canvas = self.tk.Canvas(holder, highlightthickness=0)
        scrollbar = self.ttk.Scrollbar(holder, orient="vertical", command=canvas.yview)
        rows = self.ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        def scroll(event):
            units = mousewheel_scroll_units(
                delta=getattr(event, "delta", 0), button=getattr(event, "num", 0),
                platform_name=self.root.tk.call("tk", "windowingsystem"),
            )
            if units:
                canvas.yview_scroll(units, "units")
            return "break"
        for binding in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.window.bind(binding, scroll)
        return rows

    def _add_stage(self, parent, engine: str, stage: str, selected, multiple: bool) -> None:
        row = self.ttk.Frame(parent)
        row.pack(fill="x", pady=(10, 2))
        heading = TEST_STAGE_LABELS.get(stage, stage)
        if multiple:
            heading = f"{heading} — {engine}"
        self.ttk.Label(row, text=heading, font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.stage_vars[(engine, stage)] = self.tk.StringVar(value="○ Queued")
        self.ttk.Label(row, textvariable=self.stage_vars[(engine, stage)]).pack(side="right")
        for entry in self._stage_entries(stage, selected):
            model_row = self.ttk.Frame(parent)
            model_row.pack(fill="x", padx=(14, 0), pady=2)
            self.ttk.Label(model_row, text=entry.label).pack(side="left", fill="x", expand=True)
            variable = self.tk.StringVar(value="○ Queued")
            self.model_vars[(engine, stage, entry.value)] = variable
            self.ttk.Label(model_row, textvariable=variable).pack(side="right")

    @staticmethod
    def _stage_entries(stage: str, entries) -> list:
        kinds = (
            {"embedding"} if stage == "emb" else {"image"} if stage == "img"
            else {"llm", "custom"} if stage in LLM_BACKED_TESTS else set()
        )
        return [entry for entry in entries if entry.kind in kinds]

    @staticmethod
    def _model_backed_stage(stage: str) -> bool:
        return stage == "emb" or stage == "img" or stage in LLM_BACKED_TESTS
