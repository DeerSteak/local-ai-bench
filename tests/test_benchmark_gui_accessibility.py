from scripts.app.benchmark_gui_accessibility import (
    clear_notebook_focus, configure_explicit_tab_order, focus_in_order, focus_relative,
    invoke_control, mark_notebook_focus,
)


class FakeWidget:
    def __init__(self, *, disabled=False):
        self.disabled = disabled
        self.invocations = 0
        self.next: FakeWidget | None = None
        self.previous: FakeWidget | None = None
        self.focused = False

    def tk_focusNext(self):
        return self.next

    def tk_focusPrev(self):
        return self.previous

    def focus_set(self):
        self.focused = True

    def instate(self, _states):
        return self.disabled

    def invoke(self):
        self.invocations += 1


def test_focus_relative_moves_forward_and_backward():
    current = FakeWidget()
    current.next = FakeWidget()
    current.previous = FakeWidget()

    assert focus_relative(current, reverse=False) == "break"
    assert current.next is not None
    assert current.next.focused
    assert focus_relative(current, reverse=True) == "break"
    assert current.previous is not None
    assert current.previous.focused


def test_invoke_control_activates_enabled_control_only():
    enabled = FakeWidget()
    disabled = FakeWidget(disabled=True)

    assert invoke_control(enabled) == "break"
    assert invoke_control(disabled) == "break"
    assert enabled.invocations == 1
    assert disabled.invocations == 0


class FakeNotebook:
    def __init__(self):
        self.tabs = {"tab-1": "Configuration"}
        self._keyboard_focus_tab = None

    def select(self):
        return "tab-1"

    def tab(self, selected, option=None, **kwargs):
        if "text" in kwargs:
            self.tabs[selected] = kwargs["text"]
        return self.tabs[selected] if option == "text" else None


def test_notebook_focus_marker_is_added_and_removed():
    notebook = FakeNotebook()

    mark_notebook_focus(notebook)
    assert notebook.tabs["tab-1"] == "▶ Configuration"

    clear_notebook_focus(notebook)
    assert notebook.tabs["tab-1"] == "Configuration"


class OrderedWidget(FakeWidget):
    def __init__(self, *, visible=True, disabled=False):
        super().__init__(disabled=disabled)
        self.visible = visible
        self.options = {}
        self.bindings = {}

    def winfo_viewable(self):
        return self.visible

    def configure(self, **options):
        self.options.update(options)

    def bind(self, sequence, callback):
        self.bindings[sequence] = callback


def test_explicit_tab_order_skips_hidden_and_disabled_controls():
    first = OrderedWidget()
    hidden = OrderedWidget(visible=False)
    disabled = OrderedWidget(disabled=True)
    last = OrderedWidget()

    assert focus_in_order([first, hidden, disabled, last], first, reverse=False) == "break"
    assert last.focused
    last.focused = False
    assert focus_in_order([first, hidden, disabled, last], first, reverse=True) == "break"
    assert last.focused


def test_explicit_tab_order_marks_and_binds_every_control():
    widgets = [OrderedWidget(), OrderedWidget()]

    configure_explicit_tab_order(widgets)

    assert all(widget.options == {"takefocus": True} for widget in widgets)
    assert all(set(widget.bindings) == {"<Tab>", "<Shift-Tab>", "<ISO_Left_Tab>"}
               for widget in widgets)
