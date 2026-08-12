"""llama.cpp runtime discovery and installation for setup."""

import json
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path

from scripts.runtime import hardware
from scripts.runtime.llamacpp_tools import cuda_architecture, find_llamacpp_tool, find_nvcc
from scripts.setup.archive_safety import safe_extract_zip
from scripts.setup.resumable_download import download_file
from scripts.setup.runtime_update import (
    fetch_llamacpp_release, llamacpp_clone_command, llamacpp_source_release,
    update_macos_llamacpp,
)


def find_tool(name: str, runtime_dir: Path, platform_name: str) -> str | None:
    return find_llamacpp_tool(
        name, vendored_dir=runtime_dir, platform_name=platform_name, which_fn=shutil.which,
    )


def install_windows(runtime_dir: Path, download_dir: Path, max_cuda_version: str | None,
                    *, info, warn, fail, ok) -> bool:
    info("Fetching latest llama.cpp release info ...")
    try:
        request = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            release = json.load(response)
        tag = release["tag_name"]
    except Exception as exc:
        fail(f"Could not fetch llama.cpp release info: {exc}")
        return False
    cuda_pair = hardware.select_cuda_release_assets(release["assets"], max_cuda_version)
    if cuda_pair is not None:
        binary, runtime, cuda_version = cuda_pair
        label, assets = f"CUDA {cuda_version}", [binary, runtime]
    else:
        vulkan = next(
            (asset for asset in release["assets"]
             if "win-vulkan-x64" in asset["name"].lower()
             and asset["name"].endswith(".zip")), None,
        )
        if vulkan is None:
            fail("No Windows Vulkan build found in the latest llama.cpp release")
            return False
        label, assets = "Vulkan", [vulkan]
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
            nvidia: bool, rocm: bool, compute_capability: str | None,
            max_cuda_version: str | None, info, warn, fail, ok) -> bool:
    if platform_name == "Darwin":
        info("Downloading the latest official llama.cpp macOS release ...")
        result = update_macos_llamacpp(runtime_dir, platform.machine())
        if not result.success:
            fail(result.detail)
        return result.success
    if platform_name == "Windows":
        return install_windows(
            runtime_dir, download_dir, max_cuda_version,
            info=info, warn=warn, fail=fail, ok=ok,
        )
    if platform_name != "Linux":
        return False
    if not shutil.which("git") or not shutil.which("cmake"):
        fail("git and cmake are required to build llama.cpp from source")
        return False
    flags = []
    if nvidia:
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
        info("Building with ROCm/HIP support ...")
        flags.append("-DGGML_HIP=ON")
    else:
        info("No GPU backend detected — building CPU-only ...")
    if runtime_dir.exists():
        info("Updating existing llama.cpp checkout ...")
        if subprocess.run(["git", "pull"], cwd=str(runtime_dir)).returncode != 0:
            warn("git pull failed — building from the existing checkout as-is")
    else:
        info("Cloning llama.cpp ...")
        try:
            tag, build_number = llamacpp_source_release(fetch_llamacpp_release())
        except Exception as exc:
            fail(f"Could not resolve the latest llama.cpp source release: {exc}")
            return False
        if subprocess.run(llamacpp_clone_command(runtime_dir, tag)).returncode != 0:
            fail("git clone failed")
            return False
        flags.append(f"-DLLAMA_BUILD_NUMBER={build_number}")
    build_dir = runtime_dir / "build"
    info(f"Configuring build ({' '.join(flags) or 'CPU-only'}) ...")
    if subprocess.run(["cmake", "-B", str(build_dir), "-S", str(runtime_dir), *flags]).returncode:
        fail("cmake configure failed")
        return False
    info("Building llama-server, llama-bench, and llama-batched-bench ...")
    command = [
        "cmake", "--build", str(build_dir), "--target", "llama-server",
        "--target", "llama-bench", "--target", "llama-batched-bench",
        "--config", "Release", "-j",
    ]
    if subprocess.run(command).returncode:
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
