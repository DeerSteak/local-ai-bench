"""Result History screen layout and widget ownership."""

from dataclasses import dataclass
from typing import Any


HISTORY_FOREGROUND = "#17202a"
HISTORY_EVEN_BACKGROUND = "#ffffff"
HISTORY_ODD_BACKGROUND = "#e8f1fa"
HISTORY_HEADING_BACKGROUND = "#dce8f3"
HISTORY_SELECTED_BACKGROUND = "#245b85"


def toggle_focused_history_item(tree) -> str:
    item = tree.focus()
    if not item:
        children = tree.get_children()
        if not children:
            return "break"
        item = children[0]
        tree.focus(item)
    if item in tree.selection():
        tree.selection_remove(item)
    else:
        tree.selection_add(item)
    tree.see(item)
    return "break"


def extend_history_selection(tree, direction: int) -> str:
    children = list(tree.get_children())
    if not children:
        return "break"
    current = tree.focus()
    index = children.index(current) if current in children else 0
    target = children[max(0, min(len(children) - 1, index + direction))]
    tree.focus(target)
    tree.selection_add(target)
    tree.see(target)
    return "break"


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
        frame, columns=(
            "date", "system", "status", "engine", "backend", "mtp", "profile", "models",
        ),
        show="headings", selectmode="extended", style="History.Treeview", takefocus=True,
    )
    style = ttk.Style(tree)
    style.configure(
        "History.Treeview", background=HISTORY_EVEN_BACKGROUND,
        fieldbackground=HISTORY_EVEN_BACKGROUND, foreground=HISTORY_FOREGROUND,
    )
    style.configure(
        "History.Treeview.Heading", background=HISTORY_HEADING_BACKGROUND,
        foreground=HISTORY_FOREGROUND,
    )
    style.map(
        "History.Treeview.Heading",
        background=[("active", HISTORY_ODD_BACKGROUND), ("!disabled", HISTORY_HEADING_BACKGROUND)],
        foreground=[("!disabled", HISTORY_FOREGROUND)],
    )
    style.map(
        "History.Treeview",
        background=[("selected", HISTORY_SELECTED_BACKGROUND)],
        foreground=[("selected", "#ffffff")],
    )
    tree.tag_configure(
        "history_even", background=HISTORY_EVEN_BACKGROUND, foreground=HISTORY_FOREGROUND,
    )
    tree.tag_configure(
        "history_odd", background=HISTORY_ODD_BACKGROUND, foreground=HISTORY_FOREGROUND,
    )
    for column, label, width in (
        ("date", "Started", 170), ("system", "System", 190), ("status", "Status", 95),
        ("engine", "Engine", 85), ("backend", "Backend", 80), ("mtp", "MTP", 55),
        ("profile", "Profile", 110), ("models", "Models", 70),
    ):
        tree.heading(column, text=label)
        tree.column(column, width=width, anchor="w")
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.bind("<space>", lambda _event: toggle_focused_history_item(tree))
    tree.bind("<Shift-Up>", lambda _event: extend_history_selection(tree, -1))
    tree.bind("<Shift-Down>", lambda _event: extend_history_selection(tree, 1))
    tree.grid(row=2, column=0, sticky="nsew")
    scrollbar.grid(row=2, column=1, sticky="ns")
    actions = ttk.Frame(frame)
    actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    review_actions = ttk.Frame(actions)
    review_actions.pack(fill="x")
    recovery_actions = ttk.Frame(actions)
    recovery_actions.pack(fill="x", pady=(8, 0))
    message = tk.StringVar(
        value="History has not been loaded. Keyboard: Shift+Up/Down extends selection; Space toggles a row.",
    )
    ttk.Label(frame, textvariable=message).grid(row=4, column=0, sticky="w", pady=(8, 0))
    return HistoryScreen(
        frame, filters, query, status_filter, engine_filter, engine_combo, tree,
        review_actions, recovery_actions, message,
    )
