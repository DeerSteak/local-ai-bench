"""Transactional update helpers for app-managed inference runtimes."""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import uuid

from scripts.setup.vllm_install import VllmSupport, install_vllm


@dataclass(frozen=True)
class RuntimeUpdateResult:
    success: bool
    detail: str
    version: str | None = None


def vllm_executable(venv_dir: Path, os_name: str = os.name) -> Path:
    subdir = "Scripts" if os_name == "nt" else "bin"
    suffix = ".exe" if os_name == "nt" else ""
    return venv_dir / subdir / f"vllm{suffix}"


def validate_vllm_environment(venv_dir: Path, *, run=subprocess.run,
                              os_name: str = os.name) -> RuntimeUpdateResult:
    executable = vllm_executable(venv_dir, os_name)
    if not executable.is_file():
        return RuntimeUpdateResult(False, f"Staged vLLM executable is missing: {executable}")
    try:
        result = run([str(executable), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeUpdateResult(False, f"Staged vLLM validation failed: {exc}")
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0 or not output:
        return RuntimeUpdateResult(False, output or "Staged vLLM validation returned no version.")
    return RuntimeUpdateResult(True, "Staged vLLM environment validated.", output.splitlines()[0])


def update_managed_vllm(support: VllmSupport, target: Path, *, log=print,
                        installer=install_vllm, run=subprocess.run,
                        replace=os.replace, remove=shutil.rmtree,
                        token_factory=lambda: uuid.uuid4().hex) -> RuntimeUpdateResult:
    """Build and validate a sibling venv, then swap it in with rollback."""
    target = Path(target)
    if support.method is None:
        return RuntimeUpdateResult(False, support.reason)
    if not target.is_dir():
        return RuntimeUpdateResult(False, f"Managed vLLM environment does not exist: {target}")
    token = token_factory()
    staged = target.with_name(f".{target.name}-update-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    try:
        if not installer(support, log=log, run=run, venv_dir=staged):
            return RuntimeUpdateResult(False, "The staged vLLM installation failed.")
        validation = validate_vllm_environment(staged, run=run)
        if not validation.success:
            return validation
        replace(target, backup)
        try:
            replace(staged, target)
        except Exception as exc:
            try:
                replace(backup, target)
            except Exception as rollback_exc:
                return RuntimeUpdateResult(
                    False, f"vLLM swap and rollback failed: {exc}; rollback: {rollback_exc}",
                )
            return RuntimeUpdateResult(
                False, f"vLLM update failed; the prior environment was preserved: {exc}",
            )
        try:
            remove(backup)
        except OSError as exc:
            return RuntimeUpdateResult(
                True, f"vLLM updated, but its backup remains at {backup}: {exc}", validation.version,
            )
        return RuntimeUpdateResult(True, "vLLM updated successfully.", validation.version)
    except Exception as exc:
        return RuntimeUpdateResult(False, f"vLLM update failed; the prior environment was preserved: {exc}")
    finally:
        if staged.exists():
            try:
                remove(staged)
            except OSError:
                pass
