"""Terminal presentation and prompts for interactive setup."""

from scripts.runtime import config


GREEN, YELLOW, RED, CYAN, RESET, BOLD = (
    config.GREEN, config.YELLOW, config.RED, config.CYAN, config.RESET, config.BOLD,
)


def ok(message: str) -> None:
    print(f"  {GREEN}✓{RESET}  {message}")


def warn(message: str) -> None:
    print(f"  {YELLOW}!{RESET}  {message}")


def fail(message: str) -> None:
    print(f"  {RED}✗{RESET}  {message}")


def info(message: str) -> None:
    print(f"  {CYAN}→{RESET}  {message}")


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}\n" + "─" * 50)


def link(url: str, text: str | None = None) -> str:
    return f"\033]8;;{url}\033\\{text or url}\033]8;;\033\\"


def confirm(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    try:
        reply = input(f"  {CYAN}{prompt} {hint}{RESET} ").strip().lower()
    except EOFError:
        print()
        return default
    if not reply:
        return default
    return reply in ("y", "yes")
