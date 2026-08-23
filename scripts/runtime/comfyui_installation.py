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


def _model_paths_section(name: str, models_dir: Path) -> str:
    return (
        f"{name}:\n"
        f"  base_path: {json.dumps(str(Path(models_dir).resolve()))}\n"
        "  checkpoints: checkpoints\n"
        "  diffusion_models: diffusion_models\n"
        "  clip: clip\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n"
    )


def write_extra_model_paths(path: Path, models_dir: Path,
                            legacy_models_dir: Path | None = None) -> None:
    """Write ComfyUI configuration for benchmark-owned model directories, plus the
    pre-4.1 location when models are still there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _model_paths_section("local_ai_bench", models_dir)
    if legacy_models_dir is not None:
        text += _model_paths_section("local_ai_bench_legacy", legacy_models_dir)
    path.write_text(text, encoding="utf-8")


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
        "  diffusion_models: diffusion_models\n"
        "  clip: clip\n"
        "  text_encoders: text_encoders\n"
        "  vae: vae\n"
        f"{end}\n"
    )
    separator = "\n" if existing and not existing.endswith("\n") else ""
    config_path.write_text(existing + separator + block, encoding="utf-8")
    return config_path


# Image models lived under <ComfyUI>/models before 4.1 moved them to models/comfyui.
# Both are searched, so an upgrade reuses them in place instead of re-downloading.
LEGACY_MODEL_SUBDIRS = ("checkpoints", "diffusion_models", "clip", "vae", "text_encoders")


def legacy_models_dir_with_assets(comfyui_dir: Path | None, has_files_fn=None) -> Path | None:
    """<ComfyUI>/models when it still holds image assets, else None."""
    has_files_fn = has_files_fn or (
        lambda path: path.is_dir() and any(item.is_file() for item in path.iterdir()))
    if not comfyui_dir:
        return None
    legacy = Path(comfyui_dir) / "models"
    return legacy if any(has_files_fn(legacy / sub) for sub in LEGACY_MODEL_SUBDIRS) else None


def image_asset_dirs(managed_models_dir: Path, subdir: str,
                     comfyui_dir: Path | None = None) -> list[Path]:
    """Directories holding one kind of image asset, managed location first."""
    dirs = [Path(managed_models_dir) / subdir]
    if comfyui_dir and subdir in LEGACY_MODEL_SUBDIRS:
        dirs.append(Path(comfyui_dir) / "models" / subdir)
    return dirs


def find_image_asset(name: str, managed_models_dir: Path, subdir: str,
                     comfyui_dir: Path | None = None, exists_fn=None) -> Path | None:
    """Existing path of one image asset across both locations, or None."""
    exists_fn = exists_fn or (lambda path: path.exists())
    for directory in image_asset_dirs(managed_models_dir, subdir, comfyui_dir):
        candidate = directory / name
        if exists_fn(candidate):
            return candidate
    return None


COMFYUI_LOADER_MODEL_INPUTS = {
    "CheckpointLoaderSimple": "ckpt_name",
    "UNETLoader": "unet_name",
}


def checkpoint_names_from_object_info(data: dict, loader: str) -> set[str]:
    """Extract checkpoint choices from ComfyUI's object-info response."""
    input_name = COMFYUI_LOADER_MODEL_INPUTS.get(loader)
    if input_name is None:
        raise ValueError(f"unsupported ComfyUI checkpoint loader: {loader}")
    try:
        choices = data[loader]["input"]["required"][input_name][0]
    except (KeyError, IndexError, TypeError):
        return set()
    return {str(choice) for choice in choices} if isinstance(choices, list) else set()


def managed_checkpoints_visible(available: set[str], managed: set[str]) -> bool:
    """Return whether a running server sees at least one managed checkpoint."""
    return not managed or bool(available & managed)
