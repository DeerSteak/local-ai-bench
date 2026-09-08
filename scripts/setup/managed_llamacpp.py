"""Install compatible official binaries, or build the same release in staging."""

import os
import platform
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from scripts.runtime import hardware
from scripts.setup.archive_safety import safe_extract_tar, safe_extract_zip
from scripts.setup.directory_transaction import swap_staged_directory
from scripts.setup.resumable_download import download_file
from scripts.setup.runtime_update import (
    RuntimeUpdateResult, fetch_llamacpp_release, llamacpp_source_release,
    rebuild_managed_llamacpp, select_macos_llamacpp_asset, validate_llamacpp_build,
)
from scripts.setup.intel_xpu_install import oneapi_environment


def ubuntu_compatible(release: dict) -> bool:
    return release.get("ID") == "ubuntu" or "ubuntu" in release.get("ID_LIKE", "").split()


def rocm_archive_warning(release: dict, system: str, machine: str, backend: str,
                         installed, os_release: dict) -> str | None:
    if system != "Linux" or backend != "rocm" or not installed or not ubuntu_compatible(os_release):
        return None
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine.lower())
    versions = []
    for asset in release.get("assets", []):
        match = re.search(rf"-bin-ubuntu-rocm-(\d+)\.(\d+)-{arch}\.tar\.gz$",
                          str(asset.get("name", "")))
        if match:
            versions.append(tuple(map(int, match.groups())))
    if not versions or installed >= min(versions):
        return None
    current = ".".join(map(str, installed))
    required = ".".join(map(str, min(versions)))
    return (f"Installed ROCm {current} is older than the ROCm {required} Ubuntu archives "
            "in this llama.cpp release. Building against your installed ROCm instead. "
            "To use a prebuilt archive, use a matching ROCm version supported by your OS and GPU; "
            "Local AI Bench will not upgrade ROCm automatically.")


def release_assets(release: dict, system: str, machine: str, backend: str, *,
                   max_cuda_version=None, rocm_version=None, os_release=None) -> list[dict]:
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine.lower())
    if arch is None:
        return []
    if system == "Linux" and not ubuntu_compatible(os_release or {}):
        return []
    assets = release.get("assets", [])
    if system == "Darwin":
        asset = select_macos_llamacpp_asset(release, machine) if backend == "metal" else None
        return [asset] if asset else []
    if system == "Windows" and backend == "cuda":
        if arch != "x64":
            return []
        pair = hardware.select_cuda_release_assets(assets, max_cuda_version)
        return list(pair[:2]) if pair else []
    variant = {"cpu": "", "vulkan": "vulkan-", "xpu": "sycl-fp32-" if system == "Linux" else "sycl-"}.get(backend)
    if backend == "rocm" and rocm_version:
        variant = f"rocm-{rocm_version[0]}.{rocm_version[1]}-"
    if variant is None or system not in {"Linux", "Windows"}:
        return []
    suffix = f"-bin-{'ubuntu' if system == 'Linux' else 'win'}-{variant}{arch}.{'tar.gz' if system == 'Linux' else 'zip'}"
    return [asset for asset in assets if str(asset.get("name", "")).endswith(suffix)][:1]


def install_managed_llamacpp(target: Path, system: str, machine: str, backend: str, *,
                             release_fetcher=fetch_llamacpp_release,
                             max_cuda_version=None, rocm_version=None, os_release=None,
                             log=print, warn=None, control=None, run=subprocess.run,
                             downloader=download_file, source_builder=rebuild_managed_llamacpp,
                             validator=validate_llamacpp_build, token_factory=lambda: uuid.uuid4().hex):
    """Validate complete toolsets before replacing an existing managed runtime."""
    target = Path(target)
    if target.is_symlink() or system not in {"Linux", "Windows", "Darwin"}:
        return RuntimeUpdateResult(False, "Unsupported platform or symlinked managed runtime directory.")
    if control and control.cancelled:
        return RuntimeUpdateResult(False, "Runtime installation cancelled; prior runtime preserved.")
    if system == "Linux" and os_release is None:
        try:
            os_release = platform.freedesktop_os_release()
        except OSError:
            os_release = {}
    active_run = control.run if control else run
    token = token_factory()
    staged = target.with_name(f".{target.name}-release-{token}")
    downloads = target.with_name(f".{target.name}-downloads-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    env = oneapi_environment() if backend == "xpu" else None
    def validate(path):
        return validator(path, required_backend=backend, env=env, run=active_run)
    try:
        release = release_fetcher()
        tag, _ = llamacpp_source_release(release)
        warning = rocm_archive_warning(release, system, machine, backend, rocm_version, os_release or {})
        if warning:
            (warn or (lambda message: log(f"Warning: {message}")))(warning)
        assets = release_assets(release, system, machine, backend,
                                max_cuda_version=max_cuda_version, rocm_version=rocm_version, os_release=os_release)
        validation = None
        if assets:
            try:
                downloads.mkdir(parents=True)
                for asset in assets:
                    if control and control.cancelled:
                        return RuntimeUpdateResult(False, "Runtime installation cancelled; prior runtime preserved.")
                    name, url, size = asset['name'], asset['browser_download_url'], asset['size']
                    if (not isinstance(name, str) or Path(name).name != name or '\\' in name
                            or not isinstance(url, str) or not isinstance(size, int) or size <= 0):
                        raise ValueError("Invalid release asset metadata")
                    log(f"Downloading {name} ...")
                    kwargs: dict[str, object] = {"expected_size": size}
                    if control:
                        kwargs['cancel_check'] = lambda: control.cancelled
                    archive = downloader(url, downloads / name, **kwargs)
                    (safe_extract_zip if name.endswith('.zip') else safe_extract_tar)(archive, staged)
                validation = validate(staged)
                if not validation.success:
                    log(f"Release archive is not usable: {validation.detail}")
            except Exception as exc:
                log(f"Release archive could not be used: {exc}")
        if control and control.cancelled:
            return RuntimeUpdateResult(False, "Runtime installation cancelled; prior runtime preserved.")
        if validation is None or not validation.success:
            log(f"Building {tag} from source for {system}/{machine}/{backend} ...")
            if staged.exists():
                shutil.rmtree(staged)
            staged.mkdir(parents=True)
            validation = source_builder(staged, backend, release_fetcher=lambda: release,
                                        control=control, log=log, run=run,
                                        os_name="nt" if system == "Windows" else "posix")
            if not validation.success:
                return validation
            validation = validate(staged)
        if not validation.success:
            return validation
        if control and control.cancelled:
            return RuntimeUpdateResult(False, "Runtime installation cancelled; prior runtime preserved.")
        outcome = swap_staged_directory(target, staged, backup, had_target=target.is_dir(),
                                        validate=validate, replace=os.replace, remove=shutil.rmtree)
        final = outcome.validation
        detail = f"llama.cpp {tag} installed for {backend}."
        if outcome.backup_cleanup_error:
            detail += f" Previous runtime backup remains at {backup}: {outcome.backup_cleanup_error}"
        return RuntimeUpdateResult(True, detail, final.version if final else validation.version)
    except Exception as exc:
        return RuntimeUpdateResult(False, f"llama.cpp installation failed: {exc}")
    finally:
        for path in (staged, downloads):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
