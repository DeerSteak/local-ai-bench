"""llama.cpp runtime discovery and installation for setup."""

import platform
import shutil
import subprocess
from pathlib import Path

from scripts.runtime.llamacpp_tools import (
    cuda_architecture, find_llamacpp_tool, find_nvcc, llamacpp_backend_error,
    llamacpp_backend_mismatch, managed_llamacpp_tools,
)
from scripts.setup.archive_safety import safe_extract_zip
from scripts.setup.intel_xpu_install import oneapi_environment
from scripts.setup.resumable_download import download_file
from scripts.setup.runtime_update import (
    fetch_latest_llamacpp_source_tag, fetch_llamacpp_release, fetch_llamacpp_release_tag,
    llamacpp_build_parallel_args, llamacpp_clone_command, llamacpp_source_release,
    select_windows_llamacpp_release, update_macos_llamacpp, update_windows_llamacpp,
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
    release_fetcher = (lambda: fetch_llamacpp_release_tag(version)) if version else None
    if platform_name == "Darwin":
        if vulkan:
            fail("The managed Vulkan llama.cpp runtime is available only on Windows and Linux")
            return False
        label = version or "latest"
        info(f"Downloading the {label} official llama.cpp macOS release ...")
        if release_fetcher:
            result = update_macos_llamacpp(
                runtime_dir, platform.machine(), release_fetcher=release_fetcher,
            )
        else:
            result = update_macos_llamacpp(runtime_dir, platform.machine())
        if not result.success:
            fail(result.detail)
        return result.success
    if platform_name == "Windows":
        return install_windows(
            runtime_dir, download_dir, max_cuda_version,
            intel_xpu=intel_xpu, vulkan=vulkan,
            info=info, warn=warn, fail=fail, ok=ok, release_fetcher=release_fetcher,
        )
    if platform_name != "Linux":
        return False
    if not shutil.which("git") or not shutil.which("cmake"):
        fail("git and cmake are required to build llama.cpp from source")
        return False
    flags = []
    build_env = None
    backend = "cpu"
    if vulkan:
        backend = "vulkan"
        info("Building with Vulkan support ...")
        flags.append("-DGGML_VULKAN=ON")
    elif nvidia:
        backend = "cuda"
        nvcc = find_nvcc()
        if nvcc:
            info(f"Building with CUDA support ({nvcc}) ...")
            flags += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}"]
            architecture = cuda_architecture(compute_capability)
            if architecture:
                flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
            else:
                warn("Could not read this GPU's compute capability")
        else:
            warn("NVIDIA GPU detected but the CUDA toolkit is missing; building CPU-only")
    elif rocm:
        backend = "rocm"
        info("Building with ROCm/HIP support ...")
        flags.append("-DGGML_HIP=ON")
    elif intel_xpu:
        backend = "xpu"
        build_env = oneapi_environment()
        if build_env is None:
            fail("Intel oneAPI environment is unavailable; SYCL llama.cpp cannot be built")
            return False
        info("Building with Intel oneAPI/SYCL support ...")
        flags += [
            "-DGGML_SYCL=ON", "-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx",
        ]
    else:
        info("No GPU backend detected — building CPU-only ...")
    if runtime_dir.exists():
        info("Updating existing llama.cpp checkout ...")
        if subprocess.run(["git", "pull"], cwd=str(runtime_dir)).returncode != 0:
            warn("git pull failed — building from the existing checkout as-is")
    else:
        info("Cloning llama.cpp ...")
        try:
            release = release_fetcher() if release_fetcher else fetch_llamacpp_release()
            tag, build_number = llamacpp_source_release(release)
        except Exception as exc:
            if release_fetcher:
                fail(f"Could not resolve the requested llama.cpp source release: {exc}")
                return False
            warn(f"Could not resolve llama.cpp through GitHub releases: {exc}")
            info("Falling back to the latest official llama.cpp Git tag ...")
            try:
                tag = fetch_latest_llamacpp_source_tag()
                build_number = tag[1:]
            except Exception as tag_exc:
                fail(f"Could not resolve the latest llama.cpp source tag: {tag_exc}")
                return False
        if subprocess.run(llamacpp_clone_command(runtime_dir, tag)).returncode != 0:
            fail("git clone failed")
            return False
        flags.append(f"-DLLAMA_BUILD_NUMBER={build_number}")
    build_dir = runtime_dir / "build"
    info(f"Configuring build ({' '.join(flags) or 'CPU-only'}) ...")
    if subprocess.run(
            ["cmake", "-B", str(build_dir), "-S", str(runtime_dir), *flags],
            env=build_env).returncode:
        fail("cmake configure failed")
        return False
    info("Building llama-server, llama-bench, and llama-batched-bench ...")
    parallel_args = llamacpp_build_parallel_args(backend)
    if backend == "xpu":
        info(f"Intel SYCL build parallelism: {parallel_args[-1]} job(s)")
    command = [
        "cmake", "--build", str(build_dir), "--target", "llama-server",
        "--target", "llama-bench", "--target", "llama-batched-bench",
        "--config", "Release", *parallel_args,
    ]
    if subprocess.run(command, env=build_env).returncode:
        fail("Build failed")
        return False
    if not any(build_dir.rglob("llama-server")):
        fail(f"Build finished but llama-server wasn't found under {build_dir}")
        return False
    if not any(build_dir.rglob("llama-bench")):
        warn(f"Build finished but llama-bench wasn't found under {build_dir}")
    if not any(build_dir.rglob("llama-batched-bench")):
        warn(f"Build finished but llama-batched-bench wasn't found under {build_dir}")
    return True
