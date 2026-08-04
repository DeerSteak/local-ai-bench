"""Resolve an existing ComfyUI installation without broad filesystem scans."""

import os
import platform
import sys
import json
import shlex
import subprocess
from pathlib import Path


def normalize_comfyui_dir(path: Path) -> Path | None:
    """Return the ComfyUI program directory for manual or portable layouts."""
    candidate = path.expanduser()
    if candidate.is_file():
        if candidate.name == "main.py":
            return candidate.parent.resolve()
        candidate = candidate.parent
    if (candidate / "main.py").is_file():
        return candidate.resolve()
    portable_child = candidate / "ComfyUI"
    if (portable_child / "main.py").is_file():
        return portable_child.resolve()
    return None


def resolve_comfyui_setup_choice(choice: str, entered_path: str = "") -> tuple[str, Path | None]:
    """Resolve the setup download/path choice for CLI and future GUI use."""
    normalized_choice = choice.strip().lower()
    if normalized_choice not in {"2", "p", "path"}:
        return "download", None
    resolved = normalize_comfyui_dir(Path(entered_path.strip())) if entered_path.strip() else None
    return ("existing", resolved) if resolved else ("invalid", None)


def common_comfyui_candidates(home: Path, platform_name: str) -> list[Path]:
    """Conventional manual and Windows portable locations, system-first."""
    candidates = [home / "ComfyUI", home / "comfyui"]
    if platform_name == "Windows":
        candidates.extend([
            home / "ComfyUI_windows_portable",
            home / "Downloads" / "ComfyUI_windows_portable",
        ])
    return candidates


def comfyui_dirs_from_commands(commands: list[str], platform_name: str) -> list[Path]:
    """Extract absolute ComfyUI directories from running-process commands."""
    found: list[Path] = []
    for command in commands:
        try:
            parts = shlex.split(command, posix=platform_name != "Windows")
        except ValueError:
            continue
        for part in parts:
            main_path = Path(part.strip('"'))
            if main_path.name == "main.py" and main_path.is_absolute():
                found.append(main_path.parent)
                break
    return found


def running_comfyui_dirs(platform_name: str | None = None) -> list[Path]:
    """Best-effort ComfyUI process discovery without requiring psutil."""
    system = platform_name or platform.system()
    try:
        if system == "Windows":
            output = subprocess.check_output([
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine",
            ], text=True, stderr=subprocess.DEVNULL)
        else:
            output = subprocess.check_output(
                ["ps", "-axo", "command="], text=True, stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.CalledProcessError):
        return []
    return comfyui_dirs_from_commands(output.splitlines(), system)


def find_comfyui_installation(
    *,
    explicit: str | Path | None = None,
    environ: dict[str, str] | None = None,
    saved_path: str | Path | None = None,
    managed_dir: Path | None = None,
    home: Path | None = None,
    platform_name: str | None = None,
    process_dirs: list[Path] | None = None,
) -> Path | None:
    """Resolve explicit, environment, saved, conventional, then managed paths."""
    env = os.environ if environ is None else environ
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if env.get("COMFYUI_DIR"):
        candidates.append(Path(env["COMFYUI_DIR"]))
    system = platform_name or platform.system()
    candidates.extend(running_comfyui_dirs(system) if process_dirs is None else process_dirs)
    if saved_path:
        candidates.append(Path(saved_path))
    candidates.extend(common_comfyui_candidates(
        home or Path.home(), system,
    ))
    if managed_dir:
        candidates.append(managed_dir)

    seen: set[Path] = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded in seen:
            continue
        seen.add(expanded)
        resolved = normalize_comfyui_dir(expanded)
        if resolved:
            return resolved
    return None


def find_comfyui_python(comfyui_dir: Path, environ: dict[str, str] | None = None) -> str:
    """Prefer the selected installation's embedded or virtual Python."""
    env = os.environ if environ is None else environ
    candidates = [
        comfyui_dir.parent / "python_embeded" / "python.exe",
        comfyui_dir / "python_env" / "python.exe",
        comfyui_dir / "venv" / "bin" / "python",
        comfyui_dir / ".venv" / "bin" / "python",
        comfyui_dir / "venv" / "Scripts" / "python.exe",
        comfyui_dir / ".venv" / "Scripts" / "python.exe",
    ]
    if env.get("VIRTUAL_ENV"):
        virtual_env = Path(env["VIRTUAL_ENV"])
        candidates.extend([virtual_env / "bin" / "python", virtual_env / "Scripts" / "python.exe"])
    return str(next((path for path in candidates if path.is_file()), Path(sys.executable)))


def write_extra_model_paths(path: Path, models_dir: Path) -> None:
    """Write ComfyUI configuration for benchmark-owned model directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base_path = json.dumps(str(models_dir.resolve()))
    path.write_text(
        "local_ai_bench:\n"
        f"  base_path: {base_path}\n"
        "  checkpoints: checkpoints\n"
        "  clip: clip\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n",
        encoding="utf-8",
    )


def add_managed_models_to_comfyui(comfyui_dir: Path, models_dir: Path) -> Path:
    """Add or update Local AI Bench's marked block in ComfyUI's config."""
    config_path = comfyui_dir / "extra_model_paths.yaml"
    start = "# BEGIN local-ai-bench managed models"
    end = "# END local-ai-bench managed models"
    existing = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    if start in existing and end in existing:
        before, remainder = existing.split(start, 1)
        _, after = remainder.split(end, 1)
        existing = before.rstrip() + ("\n" if before.strip() else "") + after.lstrip("\n")
    base_path = json.dumps(str(models_dir.resolve()))
    block = (
        f"{start}\n"
        "local_ai_bench_managed_models:\n"
        f"  base_path: {base_path}\n"
        "  checkpoints: checkpoints\n"
        "  clip: clip\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n"
        f"{end}\n"
    )
    separator = "\n" if existing and not existing.endswith("\n") else ""
    config_path.write_text(existing + separator + block, encoding="utf-8")
    return config_path


def checkpoint_names_from_object_info(data: dict) -> set[str]:
    """Extract checkpoint choices from ComfyUI's object-info response."""
    try:
        choices = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, IndexError, TypeError):
        return set()
    return {str(choice) for choice in choices} if isinstance(choices, list) else set()


def managed_checkpoints_visible(available: set[str], managed: set[str]) -> bool:
    """Return whether a running server sees at least one managed checkpoint."""
    return not managed or bool(available & managed)
