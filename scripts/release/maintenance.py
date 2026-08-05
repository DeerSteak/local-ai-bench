"""Previewable maintenance plans for project-owned installation state."""

import shutil
from dataclasses import dataclass
from pathlib import Path


MANAGED_NAMES = ("bench-env", "ComfyUI", "llama.cpp", "models")
PRESERVED_NAMES = ("models", "results", "local_ai_bench_config.json", "hf.txt")


@dataclass(frozen=True)
class MaintenanceAction:
    kind: str
    path: str
    exists: bool


def build_uninstall_plan(repo_root, *, remove_models=False, remove_results=False,
                         remove_credentials=False):
    """Describe only repository-owned targets; execution requires explicit confirmation."""
    root = _validated_root(repo_root)
    names = ["bench-env", "ComfyUI", "llama.cpp"]
    if remove_models:
        names.append("models")
    if remove_results:
        names.append("results")
    if remove_credentials:
        names.append("hf.txt")
    actions = [MaintenanceAction("remove", str(root / name), (root / name).exists()) for name in names]
    preserved = [name for name in PRESERVED_NAMES if name not in names]
    actions.extend(MaintenanceAction("preserve", str(root / name), (root / name).exists()) for name in preserved)
    return tuple(actions)


def execute_uninstall_plan(repo_root, plan, confirmation):
    """Remove exactly previewed project-owned targets after a typed confirmation."""
    root = _validated_root(repo_root)
    if confirmation != "REMOVE LOCAL AI BENCH COMPONENTS":
        raise ValueError("uninstall confirmation did not match")
    allowed = {root / name for name in (*MANAGED_NAMES, *PRESERVED_NAMES)}
    targets = [Path(action.path) for action in plan if action.kind == "remove"]
    for path in targets:
        if path not in allowed or path.parent != root:
            raise ValueError(f"uninstall target is outside the managed installation: {path}")
        if path.is_symlink():
            raise ValueError(f"uninstall target must not be a symbolic link: {path}")
    removed = []
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        removed.append(str(path))
    return tuple(removed)


def installation_health(repo_root):
    """Report repair inputs without installing, downloading, or modifying anything."""
    root = _validated_root(repo_root)
    checks = {
        "environment": (root / "bench-env").is_dir(),
        "requirements": (root / "requirements.txt").is_file(),
        "setup_launcher": (root / "setup.sh").is_file() or (root / "setup.bat").is_file(),
        "benchmark_launcher": (root / "run_bench.sh").is_file() or (root / "run_bench.bat").is_file(),
    }
    return {"healthy": all(checks.values()), "checks": checks}


def _validated_root(repo_root):
    root = Path(repo_root).resolve()
    if not (root / "README.md").is_file() or not (root / "scripts").is_dir():
        raise ValueError("maintenance requires a Local AI Bench repository root")
    return root
