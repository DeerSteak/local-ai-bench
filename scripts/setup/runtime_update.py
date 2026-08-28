"""Transactional update helpers for app-managed inference runtimes."""

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
from typing import Callable

import psutil

from scripts.setup.vllm_install import (
    VllmSupport, install_vllm, vllm_runtime_expectations, vllm_runtime_import_error,
)
from scripts.setup.intel_xpu_install import oneapi_environment
from scripts.setup.archive_safety import safe_extract_tar, safe_extract_zip
from scripts.setup.directory_transaction import DirectorySwapError, swap_staged_directory
from scripts.setup.resumable_download import download_file
from scripts.runtime import config, hardware
from scripts.runtime.llamacpp_tools import (
    cuda_architecture, find_nvcc, llamacpp_backend_error,
)
from scripts.setup.runtime_identity import RuntimeIdentity, parse_runtime_version, source_commit_version


LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMACPP_TARGETS = ("llama-server", "llama-bench", "llama-batched-bench")
LLAMACPP_RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"


@dataclass(frozen=True)
class RuntimeUpdateResult:
    success: bool
    detail: str
    version: str | None = None


@dataclass(frozen=True)
class WindowsLlamacppRelease:
    label: str
    assets: tuple[dict, ...]


class RuntimeUpdateControl:
    def __init__(self, output: Callable[[str], None] | None = None):
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._process = None
        self._output = output

    def emit(self, text: str | bytes) -> None:
        if self._output is not None and text:
            decoded = text.decode("utf-8", errors="replace") if isinstance(text, bytes) else text
            try:
                sys.stdout.write(decoded)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or "utf-8"
                sys.stdout.write(decoded.encode(encoding, errors="replace").decode(encoding))
            sys.stdout.flush()
            self._output(decoded)

    def log(self, text: str) -> None:
        self.emit(f"{text}\n")

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            process = self._process
        if process is not None:
            _terminate_process_tree(process)

    def track_process(self, process) -> None:
        with self._lock:
            self._process = process
        if self.cancelled:
            _terminate_process_tree(process)

    def clear_process(self, process) -> None:
        with self._lock:
            if self._process is process:
                self._process = None

    def run(self, command, **kwargs):
        if self.cancelled:
            return subprocess.CompletedProcess(command, -1, "", "update cancelled")
        timeout = kwargs.pop("timeout", None)
        original_timeout = timeout
        capture_output = kwargs.pop("capture_output", False)
        if capture_output:
            kwargs["stdout"], kwargs["stderr"] = subprocess.PIPE, subprocess.PIPE
        stream_output = self._output is not None and "stdout" not in kwargs and "stderr" not in kwargs
        if stream_output:
            kwargs["stdout"], kwargs["stderr"] = subprocess.PIPE, subprocess.STDOUT
            kwargs.setdefault("text", True)
            if kwargs["text"]:
                kwargs.setdefault("encoding", "utf-8")
                kwargs.setdefault("errors", "replace")
            kwargs.setdefault("bufsize", 1)
            env = dict(kwargs.get("env") or os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            kwargs["env"] = env
        if os.name == "nt":
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs.setdefault("start_new_session", True)
        process = subprocess.Popen(command, **kwargs)
        self.track_process(process)
        output_parts = []
        reader = None
        if stream_output:
            def read_output():
                assert process.stdout is not None
                for chunk in process.stdout:
                    output_parts.append(chunk)
                    self.emit(chunk)
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
        try:
            while True:
                try:
                    if stream_output:
                        process.wait(timeout=0.2)
                        assert reader is not None
                        reader.join()
                        stdout, stderr = "".join(output_parts), None
                    else:
                        stdout, stderr = process.communicate(timeout=0.2)
                        if capture_output:
                            self.emit(stdout or "")
                            self.emit(stderr or "")
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    if self.cancelled:
                        _terminate_process_tree(process)
                    if timeout is not None:
                        timeout -= 0.2
                        if timeout <= 0:
                            _terminate_process_tree(process)
                            raise subprocess.TimeoutExpired(command, original_timeout or 0)
        finally:
            self.clear_process(process)


def llamacpp_source_release(release: dict) -> tuple[str, str]:
    tag = normalize_llamacpp_release_tag(release.get("tag_name"))
    return tag, tag[1:]


def normalize_llamacpp_release_tag(value) -> str:
    text = str(value or "").strip().lower()
    if text.isdigit():
        text = f"b{text}"
    if not text.startswith("b") or not text[1:].isdigit():
        raise ValueError("Enter a llama.cpp release tag such as b10362 or 10362.")
    return text


def llamacpp_clone_command(destination: Path, tag: str) -> list[str]:
    return ["git", "clone", "--branch", tag, "--depth", "1", LLAMACPP_REPO, str(destination)]


def latest_llamacpp_tag_from_refs(output: str) -> str:
    tags = []
    for line in output.splitlines():
        ref = line.rsplit("/", 1)[-1].strip()
        if ref.startswith("b") and ref[1:].isdigit():
            tags.append(ref)
    if not tags:
        raise ValueError("GitHub returned no llama.cpp source tags")
    return max(tags, key=lambda tag: int(tag[1:]))


def fetch_latest_llamacpp_source_tag(*, run=subprocess.run) -> str:
    result = run(
        ["git", "ls-remote", "--tags", "--refs", LLAMACPP_REPO, "b*"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "git ls-remote failed").strip()
        raise RuntimeError(detail)
    return latest_llamacpp_tag_from_refs(result.stdout or "")


def _terminate_process_tree(process) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=3)
        for item in alive:
            item.kill()
    except (psutil.Error, OSError):
        try:
            process.kill()
        except OSError:
            pass


def _cancelled(control: RuntimeUpdateControl | None) -> RuntimeUpdateResult | None:
    return RuntimeUpdateResult(False, "Runtime update cancelled; the prior runtime was preserved.") \
        if control is not None and control.cancelled else None


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


def detect_nvidia_max_cuda_version(*, run=subprocess.run) -> str | None:
    try:
        result = run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return hardware.parse_nvidia_max_cuda_version(result.stdout) if result.returncode == 0 else None


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
    if backend == "xpu":
        return [
            "-DGGML_SYCL=ON", "-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx",
        ]
    if backend == "vulkan":
        return ["-DGGML_VULKAN=ON"]
    return []


def llamacpp_build_job_count(backend: str, *, total_memory_bytes: int | None = None) -> int | None:
    if backend != "xpu":
        return None
    total = psutil.virtual_memory().total if total_memory_bytes is None else total_memory_bytes
    total_gib = total / (1024 ** 3)
    return 8 if total_gib > 60 else 4 if total_gib > 30 else 1


def llamacpp_build_parallel_args(backend: str, *,
                                 total_memory_bytes: int | None = None) -> list[str]:
    jobs = llamacpp_build_job_count(backend, total_memory_bytes=total_memory_bytes)
    return ["--parallel", str(jobs)] if jobs is not None else ["-j"]


def validate_llamacpp_build(source_dir: Path, *, required_backend: str | None = None,
                            env=None, run=subprocess.run) -> RuntimeUpdateResult:
    tools = {}
    for name in LLAMACPP_TARGETS:
        matches = [path for path in source_dir.rglob(name) if path.is_file()]
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
    backend_error = llamacpp_backend_error(
        tools["llama-server"], required_backend, env=env, run=run,
        context="staged llama.cpp build",
    )
    if backend_error:
        return RuntimeUpdateResult(False, backend_error)
    identity = RuntimeIdentity(
        "llamacpp", "app_managed", str(tools["llama-server"]),
        parse_runtime_version(output), output,
    )
    version = source_commit_version(identity, source_dir, run=run)
    return RuntimeUpdateResult(True, "Staged llama.cpp build validated.", version)


def homebrew_llamacpp_prefix(*, run=subprocess.run) -> Path | None:
    try:
        result = run(
            ["brew", "--prefix", "llama.cpp"], capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return Path(value) if result.returncode == 0 and value else None


def update_homebrew_llamacpp(location: str | Path | None, *, run=subprocess.run,
                             control: RuntimeUpdateControl | None = None) -> RuntimeUpdateResult:
    active_run = control.run if control is not None else run
    prefix = homebrew_llamacpp_prefix(run=active_run)
    if prefix is None or location is None:
        return RuntimeUpdateResult(False, "A Homebrew llama.cpp installation was not found.")
    try:
        Path(location).resolve().relative_to(prefix.resolve())
    except (OSError, ValueError):
        return RuntimeUpdateResult(False, "The active llama.cpp runtime is not owned by Homebrew.")
    env = {**os.environ, "HOMEBREW_NO_ASK": "1", "NONINTERACTIVE": "1"}
    for command in (["brew", "update"], ["brew", "upgrade", "llama.cpp"]):
        if cancelled := _cancelled(control):
            return cancelled
        try:
            result = active_run(command, env=env)
        except OSError as exc:
            return RuntimeUpdateResult(False, f"Homebrew update failed: {exc}")
        if result.returncode != 0:
            if control is not None and control.cancelled:
                return RuntimeUpdateResult(
                    False, "Homebrew update cancelled; verify the installed formula before retrying.",
                )
            return RuntimeUpdateResult(False, f"Homebrew command failed: {' '.join(command)}")
    missing = [name for name in LLAMACPP_TARGETS if not (prefix / "bin" / name).is_file()]
    if cancelled := _cancelled(control):
        return cancelled
    if missing:
        return RuntimeUpdateResult(False, f"Updated Homebrew formula is missing: {', '.join(missing)}")
    try:
        result = active_run(
            [str(prefix / "bin" / "llama-server"), "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeUpdateResult(False, f"Updated llama.cpp validation failed: {exc}")
    output = (result.stdout or result.stderr or "").strip()
    if cancelled := _cancelled(control):
        return cancelled
    if result.returncode != 0 or not output:
        return RuntimeUpdateResult(False, output or "Updated llama.cpp returned no version.")
    return RuntimeUpdateResult(True, "Homebrew llama.cpp updated successfully.", output.splitlines()[0])


def fetch_llamacpp_release(*, opener=urllib.request.urlopen) -> dict:
    releases = fetch_llamacpp_releases(opener=opener)
    if not releases:
        raise ValueError("GitHub returned no published llama.cpp build releases")
    return releases[0]


def fetch_llamacpp_releases(*, opener=urllib.request.urlopen) -> list[dict]:
    request = urllib.request.Request(
        LLAMACPP_RELEASES_URL, headers={"Accept": "application/vnd.github+json"},
    )
    with opener(request, timeout=15) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("GitHub returned invalid llama.cpp release history")
    return [
        release for release in payload
        if isinstance(release, dict) and not release.get("draft")
        and isinstance(release.get("tag_name"), str)
        and release["tag_name"].startswith("b") and release["tag_name"][1:].isdigit()
    ]


def fetch_llamacpp_release_tag(tag: str, *, opener=urllib.request.urlopen) -> dict:
    normalized = normalize_llamacpp_release_tag(tag)
    request = urllib.request.Request(
        f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{normalized}",
        headers={"Accept": "application/vnd.github+json"},
    )
    with opener(request, timeout=15) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"GitHub returned invalid metadata for {normalized}")
    llamacpp_source_release(payload)
    return payload


def select_windows_llamacpp_release(release: dict, max_cuda_version: str | None, *,
                                    intel_xpu: bool = False,
                                    vulkan: bool = False) -> WindowsLlamacppRelease | None:
    assets = release.get("assets", [])
    if intel_xpu and vulkan:
        raise ValueError("SYCL and Vulkan llama.cpp selection are mutually exclusive")
    if intel_xpu:
        sycl = next((asset for asset in assets
                     if "win-sycl-x64" in str(asset.get("name", "")).lower()
                     and str(asset.get("name", "")).endswith(".zip")), None)
        return WindowsLlamacppRelease("SYCL", (sycl,)) if sycl is not None else None
    if not vulkan:
        cuda_pair = hardware.select_cuda_release_assets(assets, max_cuda_version)
        if cuda_pair is not None:
            return WindowsLlamacppRelease(
                f"CUDA {cuda_pair[2]}", (cuda_pair[0], cuda_pair[1]),
            )
    vulkan_asset = next((asset for asset in assets
                         if "win-vulkan-x64" in str(asset.get("name", "")).lower()
                         and str(asset.get("name", "")).endswith(".zip")), None)
    return WindowsLlamacppRelease("Vulkan", (vulkan_asset,)) \
        if vulkan_asset is not None else None


def select_windows_llamacpp_assets(release: dict, max_cuda_version: str | None, *,
                                   intel_xpu: bool = False, vulkan: bool = False) -> list[dict]:
    selected = select_windows_llamacpp_release(
        release, max_cuda_version, intel_xpu=intel_xpu, vulkan=vulkan,
    )
    return list(selected.assets) if selected is not None else []


def select_macos_llamacpp_asset(release: dict, machine: str) -> dict | None:
    architecture = "arm64" if machine.lower() in {"arm64", "aarch64"} else "x64" \
        if machine.lower() in {"x86_64", "amd64"} else None
    if architecture is None:
        return None
    suffix = f"-bin-macos-{architecture}.tar.gz"
    return next((asset for asset in release.get("assets", [])
                 if str(asset.get("name", "")).endswith(suffix)), None)


def validate_windows_llamacpp(directory: Path, *, run=subprocess.run) -> RuntimeUpdateResult:
    tools = {}
    for name in LLAMACPP_TARGETS:
        matches = [path for path in directory.rglob(f"{name}.exe") if path.is_file()]
        if not matches:
            return RuntimeUpdateResult(False, f"Staged Windows release is missing {name}.exe.")
        tools[name] = matches[0]
    try:
        result = run(
            [str(tools["llama-server"]), "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeUpdateResult(False, f"Staged Windows llama.cpp validation failed: {exc}")
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0 or not output:
        return RuntimeUpdateResult(False, output or "Staged Windows llama.cpp returned no version.")
    return RuntimeUpdateResult(True, "Staged Windows llama.cpp validated.", output.splitlines()[0])


def update_macos_llamacpp(target: Path, machine: str, *,
                          release_fetcher=fetch_llamacpp_release,
                          downloader=download_file, extractor=safe_extract_tar,
                          run=subprocess.run, replace=os.replace, remove=shutil.rmtree,
                          control: RuntimeUpdateControl | None = None,
                          token_factory=lambda: uuid.uuid4().hex) -> RuntimeUpdateResult:
    """Stage an official macOS archive, then replace with final-path validation."""
    target = Path(target)
    had_target = target.is_dir()
    token = token_factory()
    staged = target.with_name(f".{target.name}-update-{token}")
    downloads = target.with_name(f".{target.name}-downloads-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    active_run = control.run if control is not None else run
    try:
        release = release_fetcher()
        asset = select_macos_llamacpp_asset(release, machine)
        if asset is None:
            return RuntimeUpdateResult(False, "The selected release has no compatible macOS asset.")
        name, url, size = asset.get("name"), asset.get("browser_download_url"), asset.get("size")
        if not isinstance(name, str) or not isinstance(url, str) or not isinstance(size, int):
            return RuntimeUpdateResult(False, "The selected macOS release asset metadata is invalid.")
        downloads.mkdir(parents=True)
        kwargs: dict[str, object] = {"expected_size": size}
        if control is not None:
            kwargs["cancel_check"] = lambda: control.cancelled
        archive = downloader(url, downloads / name, **kwargs)
        extractor(archive, staged)
        validation = validate_llamacpp_build(staged, run=active_run)
        if not validation.success:
            return validation
        try:
            outcome = swap_staged_directory(
                target, staged, backup, had_target=had_target,
                validate=lambda path: validate_llamacpp_build(path, run=active_run),
                replace=replace, remove=remove,
            )
        except DirectorySwapError as exc:
            return RuntimeUpdateResult(
                False, f"macOS llama.cpp update failed"
                f"{' and the prior release was preserved' if had_target else ''}: {exc}",
            )
        final_validation = outcome.validation
        assert final_validation is not None
        if outcome.backup_cleanup_error is not None:
            return RuntimeUpdateResult(
                True, f"llama.cpp updated, but its backup remains at {backup}: "
                f"{outcome.backup_cleanup_error}", final_validation.version,
            )
        return RuntimeUpdateResult(
            True, "macOS llama.cpp updated successfully.", final_validation.version,
        )
    except Exception as exc:
        return RuntimeUpdateResult(False, f"macOS llama.cpp update failed: {exc}")
    finally:
        for path in (staged, downloads):
            if path.exists():
                try:
                    remove(path)
                except OSError:
                    pass


def update_windows_llamacpp(target: Path, max_cuda_version: str | None, *,
                            intel_xpu: bool = False, vulkan: bool = False,
                            release_fetcher=fetch_llamacpp_release,
                            downloader=download_file, extractor=safe_extract_zip,
                            run=subprocess.run, replace=os.replace, remove=shutil.rmtree,
                            control: RuntimeUpdateControl | None = None,
                            token_factory=lambda: uuid.uuid4().hex) -> RuntimeUpdateResult:
    """Stage and validate an official Windows release, then swap with rollback."""
    target = Path(target)
    if not target.is_dir():
        return RuntimeUpdateResult(False, f"Managed llama.cpp directory does not exist: {target}")
    token = token_factory()
    staged = target.with_name(f".{target.name}-update-{token}")
    downloads = target.with_name(f".{target.name}-downloads-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    active_run = control.run if control is not None else run
    try:
        release = release_fetcher()
        assets = select_windows_llamacpp_assets(
            release, max_cuda_version, intel_xpu=intel_xpu, vulkan=vulkan,
        )
        if not assets:
            return RuntimeUpdateResult(False, "The latest release has no compatible Windows asset.")
        downloads.mkdir(parents=True)
        for asset in assets:
            if cancelled := _cancelled(control):
                return cancelled
            name, url, size = asset.get("name"), asset.get("browser_download_url"), asset.get("size")
            if not isinstance(name, str) or not isinstance(url, str) or not isinstance(size, int):
                return RuntimeUpdateResult(False, "The selected release asset metadata is invalid.")
            download_kwargs: dict[str, object] = {"expected_size": size}
            if control is not None:
                download_kwargs["cancel_check"] = lambda: control.cancelled
            archive = downloader(url, downloads / name, **download_kwargs)
            extractor(archive, staged)
        if cancelled := _cancelled(control):
            return cancelled
        validation = validate_windows_llamacpp(staged, run=active_run)
        if cancelled := _cancelled(control):
            return cancelled
        if not validation.success:
            return validation
        try:
            outcome = swap_staged_directory(
                target, staged, backup, had_target=True, replace=replace, remove=remove,
            )
        except DirectorySwapError as exc:
            if exc.rollback_error is not None:
                return RuntimeUpdateResult(
                    False, f"Windows llama.cpp swap and rollback failed: {exc}",
                )
            return RuntimeUpdateResult(
                False, f"Windows llama.cpp update failed; the prior release was preserved: {exc}",
            )
        if outcome.backup_cleanup_error is not None:
            return RuntimeUpdateResult(
                True, f"llama.cpp updated, but its backup remains at {backup}: "
                f"{outcome.backup_cleanup_error}",
                validation.version,
            )
        return RuntimeUpdateResult(True, "Windows llama.cpp updated successfully.", validation.version)
    except Exception as exc:
        if cancelled := _cancelled(control):
            return cancelled
        return RuntimeUpdateResult(False, f"Windows llama.cpp update failed: {exc}")
    finally:
        for path in (staged, downloads):
            if path.exists():
                try:
                    remove(path)
                except OSError:
                    pass


def rebuild_managed_llamacpp(target: Path, backend: str, *, log=print,
                             run=subprocess.run, replace=os.replace,
                             remove=shutil.rmtree,
                             release_fetcher=fetch_llamacpp_release,
                             control: RuntimeUpdateControl | None = None,
                             os_name: str = os.name,
                             token_factory=lambda: uuid.uuid4().hex) -> RuntimeUpdateResult:
    """Clone and build a sibling checkout, then swap it in with rollback."""
    target = Path(target)
    if not target.is_dir():
        return RuntimeUpdateResult(False, f"Managed llama.cpp checkout does not exist: {target}")
    if os_name == "nt":
        return RuntimeUpdateResult(False, "Managed Windows release updates are not available yet.")
    nvcc = find_nvcc() if backend == "cuda" else None
    if backend == "cuda" and nvcc is None:
        return RuntimeUpdateResult(False, "CUDA rebuild requires nvcc; the current runtime was preserved.")
    active_run = control.run if control is not None else run
    capability = detect_nvidia_compute_capability(run=active_run) if backend == "cuda" else None
    build_env = oneapi_environment() if backend == "xpu" else None
    if backend == "xpu" and build_env is None:
        return RuntimeUpdateResult(
            False, "Intel XPU rebuild requires the oneAPI environment; the current runtime was preserved.",
        )
    token = token_factory()
    staged = target.with_name(f".{target.name}-update-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    try:
        tag, build_number = llamacpp_source_release(release_fetcher())
        parallel_args = llamacpp_build_parallel_args(backend)
        if backend == "xpu":
            log(f"Intel SYCL build parallelism: {parallel_args[-1]} job(s)")
        commands = [
            llamacpp_clone_command(staged, tag),
            ["cmake", "-B", str(staged / "build"), "-S", str(staged),
             "-DBUILD_SHARED_LIBS=OFF",
             f"-DLLAMA_BUILD_NUMBER={build_number}",
             *llamacpp_cmake_flags(backend, nvcc=nvcc, compute_capability=capability)],
            ["cmake", "--build", str(staged / "build"),
             *sum((["--target", name] for name in LLAMACPP_TARGETS), []),
             "--config", "Release", *parallel_args],
        ]
        for command in commands:
            if cancelled := _cancelled(control):
                return cancelled
            log(f"Running: {' '.join(command)}")
            if active_run(command, env=build_env).returncode != 0:
                if cancelled := _cancelled(control):
                    return cancelled
                return RuntimeUpdateResult(False, f"llama.cpp update command failed: {command[0]}")
        validation = validate_llamacpp_build(
            staged, required_backend=backend, env=build_env, run=active_run,
        )
        if cancelled := _cancelled(control):
            return cancelled
        if not validation.success:
            return validation
        try:
            outcome = swap_staged_directory(
                target, staged, backup, had_target=True,
                validate=lambda path: validate_llamacpp_build(
                    path, required_backend=backend, env=build_env, run=active_run,
                ),
                replace=replace, remove=remove,
            )
        except DirectorySwapError as exc:
            if exc.rollback_error is not None:
                return RuntimeUpdateResult(
                    False, f"llama.cpp update and rollback failed: {exc}",
                )
            return RuntimeUpdateResult(
                False, f"llama.cpp update failed; the prior checkout was preserved: {exc}",
            )
        final_validation = outcome.validation
        assert final_validation is not None
        if outcome.backup_cleanup_error is not None:
            return RuntimeUpdateResult(
                True, f"llama.cpp rebuilt, but its backup remains at {backup}: "
                f"{outcome.backup_cleanup_error}",
                final_validation.version,
            )
        return RuntimeUpdateResult(
            True, "llama.cpp updated and rebuilt successfully.", final_validation.version,
        )
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


def validate_vllm_environment(venv_dir: Path, *, support: VllmSupport | None = None,
                              run=subprocess.run, os_name: str = os.name) -> RuntimeUpdateResult:
    executable = vllm_executable(venv_dir, os_name)
    if not executable.is_file():
        return RuntimeUpdateResult(False, f"Staged vLLM executable is missing: {executable}")
    expected_device, expected_runtime = vllm_runtime_expectations(
        support.method if support else None,
    )
    runtime_error = vllm_runtime_import_error(
        venv_dir, expected_device_type=expected_device,
        expected_runtime=expected_runtime, run=run,
    )
    if runtime_error:
        return RuntimeUpdateResult(False, f"Staged vLLM hardware validation failed: {runtime_error}")
    try:
        result = run(
            [str(executable), "--version"], capture_output=True, text=True,
            timeout=config.VLLM_COLD_IMPORT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeUpdateResult(False, f"Staged vLLM validation failed: {exc}")
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0 or not output:
        return RuntimeUpdateResult(False, output or "Staged vLLM validation returned no version.")
    return RuntimeUpdateResult(True, "Staged vLLM environment validated.", output.splitlines()[0])


def update_managed_vllm(support: VllmSupport, target: Path, *, log=print,
                        installer=install_vllm, run=subprocess.run,
                        replace=os.replace, remove=shutil.rmtree,
                        version: str | None = None,
                        control: RuntimeUpdateControl | None = None,
                        token_factory=lambda: uuid.uuid4().hex) -> RuntimeUpdateResult:
    """Validate a sibling venv, then recreate it at its final path with rollback."""
    target = Path(target)
    if support.method is None:
        return RuntimeUpdateResult(False, support.reason)
    if not target.is_dir():
        return RuntimeUpdateResult(False, f"Managed vLLM environment does not exist: {target}")
    token = token_factory()
    staged = target.with_name(f".{target.name}-update-{token}")
    backup = target.with_name(f".{target.name}-backup-{token}")
    active_run = control.run if control is not None else run
    try:
        install_kwargs = {"version": version} if version else {}
        if not installer(support, log=log, run=active_run, venv_dir=staged, **install_kwargs):
            if cancelled := _cancelled(control):
                return cancelled
            return RuntimeUpdateResult(False, "The staged vLLM installation failed.")
        if cancelled := _cancelled(control):
            return cancelled
        validation = validate_vllm_environment(staged, support=support, run=active_run)
        if cancelled := _cancelled(control):
            return cancelled
        if not validation.success:
            return validation
        remove(staged)
        replace(target, backup)
        try:
            if not installer(support, log=log, run=active_run, venv_dir=target, **install_kwargs):
                raise RuntimeError("The final vLLM installation failed.")
            final_validation = validate_vllm_environment(
                target, support=support, run=active_run,
            )
            if not final_validation.success:
                raise RuntimeError(final_validation.detail)
        except Exception as exc:
            try:
                if target.exists():
                    remove(target)
                replace(backup, target)
            except Exception as rollback_exc:
                return RuntimeUpdateResult(
                    False, f"vLLM final install and rollback failed: {exc}; rollback: {rollback_exc}",
                )
            if cancelled := _cancelled(control):
                return cancelled
            return RuntimeUpdateResult(
                False, f"vLLM update failed; the prior environment was preserved: {exc}",
            )
        try:
            remove(backup)
        except OSError as exc:
            return RuntimeUpdateResult(
                True, f"vLLM updated, but its backup remains at {backup}: {exc}",
                final_validation.version,
            )
        return RuntimeUpdateResult(True, "vLLM updated successfully.", final_validation.version)
    except Exception as exc:
        return RuntimeUpdateResult(False, f"vLLM update failed; the prior environment was preserved: {exc}")
    finally:
        if staged.exists():
            try:
                remove(staged)
            except OSError:
                pass
