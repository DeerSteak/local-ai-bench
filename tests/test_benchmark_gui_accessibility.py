from scripts.app.benchmark_gui_accessibility import focus_relative, invoke_control


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
