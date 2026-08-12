"""Result selection, run-log persistence, and dashboard launch commands."""

import re
from pathlib import Path

from scripts.runtime import config


def selected_result_paths(selected_items, item_paths: dict, *, exact: int | None = None,
                          maximum: int | None = None) -> list[Path]:
    paths = [Path(item_paths[item]).resolve() for item in selected_items if item in item_paths]
    if exact is not None and len(paths) != exact:
        noun = "result" if exact == 1 else "results"
        raise ValueError(f"Select exactly {exact} {noun} first.")
    if not paths:
        raise ValueError("Select at least one result first.")
    if maximum is not None and len(paths) > maximum:
        raise ValueError(f"Select no more than {maximum} results.")
    return paths


def run_log_path(result_path: Path) -> Path:
    result_path = Path(result_path)
    stem = result_path.stem
    suffix = stem[len("results_"):] if stem.startswith("results_") else stem
    return result_path.with_name(f"log_{suffix}.txt")


def completed_result_paths(log: str) -> list[Path]:
    paths = []
    for line in log.splitlines():
        match = re.search(r"Results saved to:\s*(.+?)\s*$", line)
        if match and "<home>" not in match.group(1).lower():
            paths.append(Path(match.group(1)).expanduser().resolve())
    return list(dict.fromkeys(paths))


def result_paths_for_log(log: str, trusted_paths: list[Path]) -> list[Path]:
    paths = trusted_paths or completed_result_paths(log)
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def record_result_path(paths: list[Path], value: str) -> list[Path]:
    resolved = Path(value).resolve()
    return paths if resolved in paths else [*paths, resolved]


def write_run_logs(log: str, result_paths: list[Path]) -> list[Path]:
    written = []
    for result_path in result_paths:
        destination = run_log_path(result_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(log, encoding="utf-8")
        written.append(destination)
    return written


def dashboard_launcher_command(result_paths: list[Path], system: str,
                               repo_root: Path = config.SCRIPT_DIR) -> list[str]:
    root = Path(repo_root).resolve()
    launcher = root / ("launch_dashboard.bat" if system == "Windows" else "launch_dashboard.sh")
    command = ["cmd", "/c", str(launcher)] if system == "Windows" else ["bash", str(launcher)]
    for result_path in result_paths:
        command.extend(("--result", str(Path(result_path).resolve())))
    return command
