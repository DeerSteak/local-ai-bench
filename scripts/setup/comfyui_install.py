"""Install or validate the ComfyUI program used by image benchmarks."""

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from scripts.setup.archive_safety import validate_7z_archive
from scripts.setup.resumable_download import download_file


PORTABLE_FILTERS = {"amd": ("amd", "AMD"), "nvidia": ("nvidia_cu", "NVIDIA"),
                    "intel": ("intel", "Intel")}


def ensure(comfyui_dir: Path, project_dir: Path, platform_name: str, gpu: str | None, *,
           compute_capability: str | None, issues: list[str], info, warn, fail, ok) -> bool:
    portable_python = comfyui_dir.parent / "python_embeded" / "python.exe"
    if comfyui_dir.exists():
        if platform_name == "Windows" and gpu in PORTABLE_FILTERS:
            label = PORTABLE_FILTERS[gpu][1]
            if not portable_python.exists():
                warn(f"ComfyUI found at {comfyui_dir} but python_embeded is missing")
                issues.append(f"Delete {comfyui_dir} and re-run setup ({label} portable required)")
                return False
            ok(f"ComfyUI found at {comfyui_dir} ({label} portable)")
            if gpu == "nvidia":
                ensure_cuda_arch(portable_python, compute_capability, issues, info, warn, fail, ok)
        else:
            ok(f"ComfyUI found at {comfyui_dir}")
        return True
    if platform_name == "Windows" and gpu in PORTABLE_FILTERS:
        asset_filter, label = PORTABLE_FILTERS[gpu]
        info(f"{label} GPU detected — downloading official ComfyUI portable build ...")
        installed = install_portable(project_dir, asset_filter, label, info, warn, fail, ok)
        if not installed:
            issues.append(f"Download ComfyUI {label} portable from ComfyUI releases")
        elif gpu == "nvidia" and portable_python.exists():
            ensure_cuda_arch(portable_python, compute_capability, issues, info, warn, fail, ok)
        return installed
    repository = "https://github.com/comfyanonymous/ComfyUI"
    info(f"Cloning ComfyUI from {repository} ...")
    if subprocess.run(["git", "clone", repository, str(comfyui_dir)]).returncode == 0:
        ok(f"ComfyUI cloned to {comfyui_dir}")
        return True
    fail("ComfyUI clone failed — check your internet connection and git install")
    issues.append(f"git clone {repository}")
    return False


def install_portable(project_dir: Path, asset_filter: str, label: str,
                     info, warn, fail, ok) -> bool:
    info("Fetching latest ComfyUI release info ...")
    try:
        request = urllib.request.Request(
            "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest",
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            release = json.load(response)
        asset = next(
            (item for item in release["assets"] if asset_filter in item["name"].lower()
             and item["name"].endswith(".7z")), None,
        )
        if asset is None:
            fail(f"No {label} portable build found in latest ComfyUI release")
            return False
    except Exception as exc:
        fail(f"Could not fetch ComfyUI release info: {exc}")
        return False
    archive = project_dir / asset["name"]
    info(f"Downloading ComfyUI {release['tag_name']} {label} portable ...")
    try:
        download_file(asset["browser_download_url"], archive, expected_size=asset["size"])
        validate_7z_archive(archive)
    except Exception as exc:
        fail(f"Download or archive validation failed: {exc}")
        archive.unlink(missing_ok=True)
        return False
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if not seven_zip:
        seven_zip = _download_7zr(project_dir, archive, info, fail, ok)
    if not seven_zip:
        return False
    result = subprocess.run(
        [seven_zip, "x", str(archive), f"-o{project_dir}", "-y"],
        capture_output=True, text=True,
    )
    if result.returncode:
        fail(f"Extraction failed:\n{result.stderr.strip()}")
        archive.unlink(missing_ok=True)
        return False
    archive.unlink()
    _flatten_portable(project_dir)
    ok(f"ComfyUI {release['tag_name']} {label} portable extracted")
    return True


def _download_7zr(project_dir: Path, archive: Path, info, fail, ok) -> str | None:
    executable = project_dir / "7zr.exe"
    if not executable.exists():
        info("Downloading 7zr.exe for extraction ...")
        try:
            download_file("https://github.com/ip7z/7zip/releases/download/26.02/7zr.exe", executable)
            ok("7zr.exe downloaded")
        except Exception as exc:
            fail(f"Could not download 7zr.exe: {exc}")
            executable.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            return None
    return str(executable)


def _flatten_portable(project_dir: Path) -> None:
    wrapper = project_dir / "ComfyUI_windows_portable"
    if not wrapper.is_dir():
        return
    for child in wrapper.iterdir():
        destination = project_dir / child.name
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        shutil.move(str(child), str(destination))
    wrapper.rmdir()


def ensure_cuda_arch(python: Path, capability: str | None, issues: list[str],
                     info, warn, fail, ok) -> None:
    if not capability:
        return
    major, minor = capability.split(".")
    architecture = f"sm_{major}{minor}"
    script = "import torch; print(','.join(torch.cuda.get_arch_list()))"
    try:
        supported = subprocess.check_output(
            [str(python), "-c", script], text=True, stderr=subprocess.DEVNULL,
        ).strip().split(",")
    except Exception as exc:
        warn(f"Could not check torch CUDA architecture support: {exc}")
        return
    if architecture in supported:
        ok(f"torch build supports {architecture}")
        return
    info(f"Reinstalling torch with cu128 support for {architecture} ...")
    command = [
        str(python), "-s", "-m", "pip", "install", "--force-reinstall", "--no-deps",
        "torch", "torchvision", "torchaudio", "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ]
    if subprocess.run(command).returncode:
        fail("torch reinstall failed")
        issues.append(" ".join(command))
