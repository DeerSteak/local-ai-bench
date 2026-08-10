"""Transactional update helpers for app-managed inference runtimes."""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import uuid

from scripts.setup.vllm_install import VllmSupport, install_vllm
from scripts.runtime.llamacpp_tools import cuda_architecture, find_nvcc


LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMACPP_TARGETS = ("llama-server", "llama-bench", "llama-batched-bench")


@dataclass(frozen=True)
class RuntimeUpdateResult:
    success: bool
    detail: str
    version: str | None = None


def detect_nvidia_compute_capability(*, run=subprocess.run) -> str | None:
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = (result.stdout or "").strip().splitlines()
    return first[0].strip() if result.returncode == 0 and first else None


def llamacpp_cmake_flags(backend: str, *, nvcc: str | None = None,
                         compute_capability: str | None = None) -> list[str]:
    if backend == "cuda":
        flags = ["-DGGML_CUDA=ON"]
        if nvcc:
            flags.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        architecture = cuda_architecture(compute_capability)
        if architecture:
            flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
        return flags
    if backend == "rocm":
        return ["-DGGML_HIP=ON"]
    return []


def validate_llamacpp_build(source_dir: Path, *, run=subprocess.run) -> RuntimeUpdateResult:
    tools = {}
    for name in LLAMACPP_TARGETS:
        matches = [path for path in (source_dir / "build").rglob(name) if path.is_file()]
        if not matches:
            return RuntimeUpdateResult(False, f"Staged llama.cpp build is missing {name}.")
        tools[name] = matches[0]
    try:
        result = run(
            [str(tools["llama-server"]), "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeUpdateResult(False, f"Staged llama.cpp validation failed: {exc}")
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0 or not output:
        return RuntimeUpdateResult(False, output or "Staged llama.cpp returned no version.")
    return RuntimeUpdateResult(True, "Staged llama.cpp build validated.", output.splitlines()[0])


def rebuild_managed_llamacpp(target: Path, backend: str, *, log=print,
                             run=subprocess.run, replace=os.replace,
                             remove=shutil.rmtree,
                             token_factory=lambda: uuid.uuid4().hex) -> RuntimeUpdateResult:
    """Clone and build a sibling checkout, then swap it in with rollback."""
    target = Path(target)
    if not target.is_dir():
        return RuntimeUpdateResult(False, f"Managed llama.cpp checkout does not exist: {target}")
    if os.name == "nt":
        return RuntimeUpdateResult(False, "Managed Windows release updates are not available yet.")
    nvcc = find_nvcc() if backend == "cuda" else None
    if backend == "cuda" and nvcc is None:
        return RuntimeUpdateResult(False, "CUDA rebuild requires nvcc; the current runtime was preserved.")
    capability = detect_nvidia_compute_capability(run=run) if backend == "cuda" else None
    token = token_factory()
    staged = target.with_name(f".{target.name}-update-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    try:
        commands = [
            ["git", "clone", "--depth", "1", LLAMACPP_REPO, str(staged)],
            ["cmake", "-B", str(staged / "build"), "-S", str(staged),
             *llamacpp_cmake_flags(backend, nvcc=nvcc, compute_capability=capability)],
            ["cmake", "--build", str(staged / "build"),
             *sum((["--target", name] for name in LLAMACPP_TARGETS), []),
             "--config", "Release", "-j"],
        ]
        for command in commands:
            log(f"Running: {' '.join(command)}")
            if run(command).returncode != 0:
                return RuntimeUpdateResult(False, f"llama.cpp update command failed: {command[0]}")
        validation = validate_llamacpp_build(staged, run=run)
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
                    False, f"llama.cpp swap and rollback failed: {exc}; rollback: {rollback_exc}",
                )
            return RuntimeUpdateResult(
                False, f"llama.cpp update failed; the prior checkout was preserved: {exc}",
            )
        try:
            remove(backup)
        except OSError as exc:
            return RuntimeUpdateResult(
                True, f"llama.cpp rebuilt, but its backup remains at {backup}: {exc}",
                validation.version,
            )
        return RuntimeUpdateResult(True, "llama.cpp updated and rebuilt successfully.", validation.version)
    except Exception as exc:
        return RuntimeUpdateResult(False, f"llama.cpp update failed: {exc}")
    finally:
        if staged.exists():
            try:
                remove(staged)
            except OSError:
                pass


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
