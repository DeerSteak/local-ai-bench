"""llama.cpp runtime discovery and installation for setup."""

import platform
import shutil
from pathlib import Path

from scripts.runtime.llamacpp_tools import (
    find_llamacpp_tool, llamacpp_backend_error,
    llamacpp_backend_mismatch, managed_llamacpp_tools,
)
from scripts.setup.archive_safety import safe_extract_zip
from scripts.setup.resumable_download import download_file
from scripts.setup.managed_llamacpp import install_managed_llamacpp
from scripts.setup.setup_discovery import rocm_version
from scripts.setup.runtime_update import (
    fetch_llamacpp_release, fetch_llamacpp_release_tag,
    select_windows_llamacpp_release, update_windows_llamacpp,
)


def find_tool(name: str, runtime_dir: Path, platform_name: str) -> str | None:
    return find_llamacpp_tool(
        name, vendored_dir=runtime_dir, platform_name=platform_name, which_fn=shutil.which,
    )


def find_tools(runtime_dir: Path, platform_name: str) -> dict[str, str | None]:
    return {
        name: find_tool(name, runtime_dir, platform_name)
        for name in ("llama-server", "llama-bench", "llama-batched-bench")
    }


def managed_toolset_ready(runtime_dir: Path, platform_name: str) -> bool:
    return bool(managed_llamacpp_tools(runtime_dir, platform_name))


def installed_toolset_error(binary: str | None, required_backend: str | None, *,
                            env=None) -> str | None:
    if binary is None:
        return "Managed llama.cpp toolset is incomplete — rerun Setup to repair it"
    return llamacpp_backend_error(binary, required_backend, env=env, context="setup")


qualification_backend_mismatch = llamacpp_backend_mismatch


def qualification_backend_error(binary: str | None, required_backend: str | None, *,
                                probe) -> str | None:
    return llamacpp_backend_error(
        binary, required_backend,
        probe=lambda value, **_kwargs: probe(value), context="qualification",
    )


def install_windows(runtime_dir: Path, download_dir: Path, max_cuda_version: str | None,
                    *, intel_xpu: bool = False, vulkan: bool = False,
                    info, warn, fail, ok,
                    release_fetcher=None) -> bool:
    info("Fetching latest llama.cpp release info ...")
    try:
        release = release_fetcher() if release_fetcher else fetch_llamacpp_release()
        tag = release["tag_name"]
    except Exception as exc:
        fail(f"Could not fetch llama.cpp release info: {exc}")
        return False
    selected = select_windows_llamacpp_release(
        release, max_cuda_version, intel_xpu=intel_xpu, vulkan=vulkan,
    )
    if selected is None:
        backend = "SYCL" if intel_xpu else "Vulkan"
        fail(f"No Windows {backend} build found in the latest llama.cpp release")
        return False
    label, assets = selected.label, selected.assets
    if runtime_dir.is_dir():
        result = update_windows_llamacpp(
            runtime_dir, max_cuda_version, intel_xpu=intel_xpu, vulkan=vulkan,
            release_fetcher=lambda: release,
        )
        if result.success:
            ok(f"llama.cpp {tag} ({label}) replaced the prior managed runtime")
        else:
            fail(result.detail)
        return result.success
    size_mb = sum(asset["size"] for asset in assets) // (1024 ** 2)
    info(f"Downloading llama.cpp {tag} ({label}, {size_mb} MB) ...")
    archives = [download_dir / asset["name"] for asset in assets]
    try:
        for asset, archive in zip(assets, archives):
            download_file(asset["browser_download_url"], archive, expected_size=asset["size"])
    except Exception as exc:
        fail(f"Download failed: {exc}")
        for archive in archives:
            archive.unlink(missing_ok=True)
        return False
    info(f"Extracting {', '.join(asset['name'] for asset in assets)} ...")
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            safe_extract_zip(archive, runtime_dir)
            archive.unlink()
    except Exception as exc:
        fail(f"Extraction failed: {exc}")
        for archive in archives:
            archive.unlink(missing_ok=True)
        return False
    if not any(runtime_dir.rglob("llama-server.exe")):
        fail(f"Extracted llama.cpp {tag} ({label}) but llama-server.exe wasn't found inside it")
        return False
    if not any(runtime_dir.rglob("llama-bench.exe")):
        warn(f"Extracted llama.cpp {tag} ({label}) without llama-bench.exe")
    if not any(runtime_dir.rglob("llama-batched-bench.exe")):
        warn(f"Extracted llama.cpp {tag} ({label}) without llama-batched-bench.exe")
    ok(f"llama.cpp {tag} ({label}) extracted to {runtime_dir}")
    return True


def install(runtime_dir: Path, download_dir: Path, platform_name: str, *,
            nvidia: bool, rocm: bool, intel_xpu: bool, compute_capability: str | None,
            max_cuda_version: str | None, info, warn, fail, ok,
            version: str | None = None, vulkan: bool = False) -> bool:
    if runtime_dir.is_symlink():
        fail(f"Managed runtime directory is an external symlink: {runtime_dir}. "
             "Remove the symlink and rerun Setup to install a project-owned copy.")
        return False
    if platform_name not in {"Darwin", "Windows", "Linux"}:
        return False
    if vulkan and platform_name == "Darwin":
        fail("The managed Vulkan llama.cpp runtime is available only on Windows and Linux")
        return False
    backend = ("vulkan" if vulkan else "metal" if platform_name == "Darwin"
               else "cuda" if nvidia else "rocm" if rocm else "xpu" if intel_xpu
               else "vulkan" if platform_name == "Windows" else "cpu")
    release_fetcher = (lambda: fetch_llamacpp_release_tag(version)) if version else fetch_llamacpp_release
    result = install_managed_llamacpp(
        runtime_dir, platform_name, platform.machine(), backend,
        release_fetcher=release_fetcher, max_cuda_version=max_cuda_version,
        rocm_version=rocm_version() if rocm else None, log=info, warn=warn,
    )
    (ok if result.success else fail)(result.detail)
    return result.success
