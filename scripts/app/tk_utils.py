"""Small shared helpers for the setup and benchmark Tk interfaces."""


def refresh_tk_layout(widget) -> None:
    """Flush layout now and once more when Tk next becomes idle."""
    widget.update_idletasks()
    widget.after_idle(widget.update_idletasks)


def mousewheel_scroll_units(*, delta: int = 0, button: int = 0,
                            platform_name: str = "") -> int:
    """Translate Tk wheel events into vertical canvas scroll units."""
    if button == 4:
        return -1
    if button == 5:
        return 1
    if not delta:
        return 0
    if platform_name == "darwin":
        return -1 if delta > 0 else 1
    units = -int(delta / 120)
    return units or (-1 if delta > 0 else 1)
