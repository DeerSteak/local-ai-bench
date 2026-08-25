"""Keyboard navigation and focus treatment for the Tk benchmark launcher."""

FOCUS_COLOR = "#0969da"


def focus_relative(widget, *, reverse: bool) -> str:
    target = widget.tk_focusPrev() if reverse else widget.tk_focusNext()
    if target is not None:
        target.focus_set()
    return "break"


def invoke_control(widget) -> str:
    if not widget.instate(("disabled",)):
        widget.invoke()
    return "break"


def _prepend_style_map(style, style_name: str, option: str, value: str) -> None:
    existing = [entry for entry in style.map(style_name, option) if "focus" not in entry[:-1]]
    style.map(style_name, **{option: [("focus", value), *existing]})


def configure_keyboard_accessibility(root, ttk) -> None:
    style = ttk.Style(root)
    for style_name in ("TButton", "Start.TButton", "TCheckbutton", "TRadiobutton"):
        style.configure(style_name, focuscolor=FOCUS_COLOR, focusthickness=3)
        _prepend_style_map(style, style_name, "foreground", FOCUS_COLOR)
    for style_name in ("TEntry", "TCombobox", "Treeview", "TNotebook.Tab"):
        style.configure(style_name, focuscolor=FOCUS_COLOR, focusthickness=3)
        _prepend_style_map(style, style_name, "bordercolor", FOCUS_COLOR)

    root.bind_all("<Tab>", lambda event: focus_relative(event.widget, reverse=False))
    root.bind_all("<Shift-Tab>", lambda event: focus_relative(event.widget, reverse=True))
    root.bind_all("<ISO_Left_Tab>", lambda event: focus_relative(event.widget, reverse=True))
    for widget_class in ("TButton", "TCheckbutton", "TRadiobutton"):
        root.bind_class(widget_class, "<space>", lambda event: invoke_control(event.widget))
