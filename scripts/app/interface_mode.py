"""Select the GUI, terminal, or noninteractive benchmark interface."""


INTERFACE_MODES = {"auto", "gui", "terminal", "none"}
SSH_ENV_KEYS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")
WSL_ENV_KEYS = ("WSL_DISTRO_NAME", "WSL_INTEROP")


def is_ssh_session(env: dict[str, str]) -> bool:
    return any(env.get(key) for key in SSH_ENV_KEYS)


def is_wsl_session(platform_name: str, env: dict[str, str]) -> bool:
    return platform_name.lower() == "linux" and any(env.get(key) for key in WSL_ENV_KEYS)


def desktop_available(platform_name: str, env: dict[str, str]) -> bool:
    platform_name = platform_name.lower()
    if platform_name == "darwin":
        return not is_ssh_session(env)
    if platform_name == "windows":
        session = env.get("SESSIONNAME", "").lower()
        return session not in {"services", "service"} and not is_ssh_session(env)
    return bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY")) and not is_ssh_session(env)


def select_interface_mode(requested: str, *, platform_name: str, env: dict[str, str],
                          stdin_is_tty: bool, gui_available: bool) -> str:
    if requested not in INTERFACE_MODES:
        raise ValueError(f"unknown interface mode: {requested}")
    if requested != "auto":
        if requested == "terminal" and not stdin_is_tty:
            raise ValueError("terminal interface requires an interactive terminal")
        if requested == "gui" and not gui_available:
            raise ValueError("GUI assets are not installed")
        return requested
    if stdin_is_tty and is_wsl_session(platform_name, env):
        return "terminal"
    if gui_available and desktop_available(platform_name, env):
        return "gui"
    if stdin_is_tty:
        return "terminal"
    return "none"
