"""Result History screen layout and widget ownership."""

from dataclasses import dataclass
from typing import Any


@dataclass
class HistoryScreen:
    frame: Any
    filters: Any
    query: Any
    status_filter: Any
    engine_filter: Any
    engine_combo: Any
    tree: Any
    review_actions: Any
    recovery_actions: Any
    message: Any


def build_history_screen(notebook, *, tk, ttk) -> HistoryScreen:
    frame = ttk.Frame(notebook, padding=18)
    notebook.add(frame, text="Result History")
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=1)
    ttk.Label(frame, text="Local result history", style="Title.TLabel").grid(
        row=0, column=0, sticky="w",
    )
    filters = ttk.Frame(frame)
    filters.grid(row=1, column=0, sticky="ew", pady=(8, 10))
    query = tk.StringVar()
    status_filter = tk.StringVar(value="all")
    engine_filter = tk.StringVar(value="all")
    ttk.Label(filters, text="Search").pack(side="left")
    ttk.Entry(filters, textvariable=query, width=26).pack(side="left", padx=(8, 14))
    ttk.Label(filters, text="Status").pack(side="left")
    ttk.Combobox(
        filters, state="readonly", width=12, textvariable=status_filter,
        values=("all", "complete", "partial", "interrupted", "failed", "running", "legacy"),
    ).pack(side="left", padx=(8, 14))
    ttk.Label(filters, text="Engine").pack(side="left")
    engine_combo = ttk.Combobox(
        filters, state="readonly", width=14, textvariable=engine_filter, values=("all",),
    )
    engine_combo.pack(side="left", padx=(8, 14))
    tree = ttk.Treeview(
        frame, columns=("date", "system", "status", "engine", "profile", "models"),
        show="headings", selectmode="extended", style="History.Treeview",
    )
    tree.tag_configure("history_even", background="#ffffff")
    tree.tag_configure("history_odd", background="#edf2f7")
    for column, label, width in (
        ("date", "Started", 170), ("system", "System", 190), ("status", "Status", 95),
        ("engine", "Engine", 95), ("profile", "Profile", 110), ("models", "Models", 70),
    ):
        tree.heading(column, text=label)
        tree.column(column, width=width, anchor="w")
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")
    actions = ttk.Frame(frame)
    actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    review_actions = ttk.Frame(actions)
    review_actions.pack(fill="x")
    recovery_actions = ttk.Frame(actions)
    recovery_actions.pack(fill="x", pady=(8, 0))
    message = tk.StringVar(value="History has not been loaded.")
    ttk.Label(frame, textvariable=message).grid(row=4, column=0, sticky="w", pady=(8, 0))
    return HistoryScreen(
        frame, filters, query, status_filter, engine_filter, engine_combo, tree,
        review_actions, recovery_actions, message,
    )
