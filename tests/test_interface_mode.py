import pytest

from scripts.app.interface_mode import desktop_available, is_ssh_session, select_interface_mode


def test_ssh_detection_accepts_common_environment_markers():
    for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        assert is_ssh_session({key: "present"})
    assert not is_ssh_session({})


@pytest.mark.parametrize("platform_name,env", [
    ("Darwin", {}),
    ("Linux", {"DISPLAY": ":0"}),
    ("Linux", {"WAYLAND_DISPLAY": "wayland-0"}),
    ("Windows", {"SESSIONNAME": "Console"}),
    ("Windows", {"SESSIONNAME": "RDP-Tcp#1"}),
])
def test_desktop_detection_accepts_local_graphical_sessions(platform_name, env):
    assert desktop_available(platform_name, env)


@pytest.mark.parametrize("platform_name,env", [
    ("Linux", {}),
    ("Linux", {"DISPLAY": ":10", "SSH_CONNECTION": "remote"}),
    ("Darwin", {"SSH_TTY": "/dev/ttys001"}),
    ("Windows", {"SESSIONNAME": "Services"}),
    ("Windows", {"SESSIONNAME": "Console", "SSH_CLIENT": "remote"}),
])
def test_desktop_detection_rejects_headless_or_ssh_sessions(platform_name, env):
    assert not desktop_available(platform_name, env)


def test_auto_prefers_gui_on_a_local_desktop():
    assert select_interface_mode(
        "auto", platform_name="Linux", env={"DISPLAY": ":0"},
        stdin_is_tty=True, gui_available=True,
    ) == "gui"


def test_auto_prefers_terminal_over_forwarded_display_in_ssh():
    assert select_interface_mode(
        "auto", platform_name="Linux",
        env={"DISPLAY": ":10", "SSH_CONNECTION": "remote"},
        stdin_is_tty=True, gui_available=True,
    ) == "terminal"


def test_auto_falls_back_to_terminal_when_gui_is_not_installed():
    assert select_interface_mode(
        "auto", platform_name="Darwin", env={},
        stdin_is_tty=True, gui_available=False,
    ) == "terminal"


def test_auto_selects_noninteractive_mode_without_desktop_or_tty():
    assert select_interface_mode(
        "auto", platform_name="Linux", env={},
        stdin_is_tty=False, gui_available=True,
    ) == "none"


def test_explicit_gui_is_allowed_over_ssh_for_port_forwarding():
    assert select_interface_mode(
        "gui", platform_name="Linux", env={"SSH_CONNECTION": "remote"},
        stdin_is_tty=True, gui_available=True,
    ) == "gui"


def test_explicit_unavailable_gui_and_noninteractive_terminal_fail_clearly():
    with pytest.raises(ValueError, match="not installed"):
        select_interface_mode(
            "gui", platform_name="Linux", env={},
            stdin_is_tty=True, gui_available=False,
        )
    with pytest.raises(ValueError, match="interactive terminal"):
        select_interface_mode(
            "terminal", platform_name="Linux", env={},
            stdin_is_tty=False, gui_available=True,
        )


def test_unknown_requested_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        select_interface_mode(
            "desktop", platform_name="Darwin", env={},
            stdin_is_tty=True, gui_available=True,
        )
