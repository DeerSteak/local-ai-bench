"""Keyboard navigation and focus treatment for the Tk benchmark launcher."""

FOCUS_COLOR = "#0969da"
DISABLED_FOREGROUND = "#6e7781"
DISABLED_BACKGROUND = "#eaeef2"


def focus_relative(widget, *, reverse: bool) -> str:
    target = widget.tk_focusPrev() if reverse else widget.tk_focusNext()
    if target is not None:
        target.focus_set()
    return "break"


def invoke_control(widget) -> str:
    if not widget.instate(("disabled",)):
        widget.invoke()
    return "break"


def mark_notebook_focus(notebook) -> None:
    clear_notebook_focus(notebook)
    selected = notebook.select()
    if not selected:
        return
    text = notebook.tab(selected, "text")
    notebook._keyboard_focus_tab = (selected, text)
    notebook.tab(selected, text=f"▶ {text}")


def clear_notebook_focus(notebook) -> None:
    marked = getattr(notebook, "_keyboard_focus_tab", None)
    if marked is None:
        return
    selected, text = marked
    try:
        notebook.tab(selected, text=text)
    except Exception:
        pass
    notebook._keyboard_focus_tab = None


def refresh_notebook_focus(notebook) -> None:
    if notebook.focus_get() == notebook:
        mark_notebook_focus(notebook)


def _prepend_style_map(style, style_name: str, option: str, state: str, value: str) -> None:
    existing = [entry for entry in style.map(style_name, option) if state not in entry[:-1]]
    style.map(style_name, **{option: [(state, value), *existing]})


def configure_keyboard_accessibility(root, ttk) -> None:
    style = ttk.Style(root)
    for style_name in ("TButton", "Start.TButton", "TCheckbutton", "TRadiobutton"):
        style.configure(style_name, focuscolor=FOCUS_COLOR, focusthickness=3)
        _prepend_style_map(style, style_name, "foreground", "focus", FOCUS_COLOR)
    for style_name in ("TEntry", "TCombobox", "Treeview", "TNotebook.Tab"):
        style.configure(style_name, focuscolor=FOCUS_COLOR, focusthickness=3)
        _prepend_style_map(style, style_name, "bordercolor", "focus", FOCUS_COLOR)
    _prepend_style_map(style, "TButton", "foreground", "disabled", DISABLED_FOREGROUND)
    _prepend_style_map(style, "TButton", "background", "disabled", DISABLED_BACKGROUND)

    root.bind_all("<Tab>", lambda event: focus_relative(event.widget, reverse=False))
    root.bind_all("<Shift-Tab>", lambda event: focus_relative(event.widget, reverse=True))
    root.bind_all("<ISO_Left_Tab>", lambda event: focus_relative(event.widget, reverse=True))
    for widget_class in ("TButton", "TCheckbutton", "TRadiobutton"):
        root.bind_class(widget_class, "<space>", lambda event: invoke_control(event.widget))
    root.bind_class("TNotebook", "<space>", lambda event: focus_relative(event.widget, reverse=False))
    root.bind_class("TNotebook", "<FocusIn>", lambda event: mark_notebook_focus(event.widget), add="+")
    root.bind_class("TNotebook", "<FocusOut>", lambda event: clear_notebook_focus(event.widget), add="+")
    root.bind_class(
        "TNotebook", "<<NotebookTabChanged>>",
        lambda event: refresh_notebook_focus(event.widget), add="+",
    )
