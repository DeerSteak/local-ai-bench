"""vLLM install support matrix and installer — see docs/setup.md's vLLM section.

Everything above `install_vllm` is pure decision logic so it can be unit tested
without running the real installer.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
import sys

from scripts.runtime import config

METAL_INSTALL_URL = "https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh"
ROCM_WHEEL_INDEX = "https://wheels.vllm.ai/rocm/"
NIGHTLY_CU130_INDEX = "https://wheels.vllm.ai/nightly/cu130"

# vLLM's own floor for the CUDA wheels; below this the kernels aren't built.
MIN_COMPUTE_CAPABILITY = 7.5
MIN_ROCM_VERSION = (6, 3)

# The ROCm and Metal builds publish CPython 3.12 wheels only; CUDA spans a range.
CUDA_PYTHON_RANGE = ((3, 10), (3, 13))
PINNED_PYTHON = (3, 12)


@dataclass(frozen=True)
class VllmSupport:
    status: str            # "supported" | "experimental" | "unsupported"
    method: str | None     # "cuda_wheel" | "rocm_wheel" | "nightly_cu130" | "metal_plugin"
    reason: str
    requires_python: tuple[int, int] | None = None

    @property
    def installable(self) -> bool:
        return self.method is not None


def parse_compute_capability(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def is_dgx_spark(machine: str, gpu_names) -> bool:
    """GB10 Grace Blackwell on ARM — stock aarch64 wheels pull CPU-only torch."""
    if str(machine).lower() not in ("aarch64", "arm64"):
        return False
    return any("gb10" in str(name).lower() for name in (gpu_names or []))


def vllm_platform_support(*, os_name: str, machine: str,
                          python_version: tuple[int, int],
                          nvidia_ok: bool = False,
                          rocm_ok: bool = False,
                          intel_gpu: bool = False,
                          gpu_names=None,
                          compute_cap: str | None = None,
                          rocm_version: tuple[int, int] | None = None) -> VllmSupport:
    """Whether setup can install vLLM here, and how. See docs/setup.md's support table."""
    if os_name == "Windows":
        return VllmSupport("unsupported", None,
                           "vLLM has no upstream Windows support — run it under WSL2 "
                           "(the Linux path applies inside WSL2) or Docker Desktop")

    if os_name == "Darwin":
        if str(machine).lower() not in ("arm64", "aarch64"):
            return VllmSupport("unsupported", None,
                               "vLLM on macOS requires Apple Silicon — there is no Intel Mac build")
        return VllmSupport("experimental", "metal_plugin",
                           "Apple Silicon uses the community-maintained vllm-metal plugin, "
                           "which installs into its own ~/.venv-vllm-metal environment",
                           requires_python=PINNED_PYTHON)

    if os_name != "Linux":
        return VllmSupport("unsupported", None, f"vLLM has no build for {os_name}")

    if nvidia_ok:
        if is_dgx_spark(machine, gpu_names):
            return VllmSupport("experimental", "nightly_cu130",
                               "DGX Spark (GB10, sm_121) is not covered by stock wheels — "
                               "the CUDA 13 nightly build is the only working path, and "
                               "plain wheels would silently install CPU-only PyTorch",
                               requires_python=PINNED_PYTHON)
        capability = parse_compute_capability(compute_cap)
        if capability is not None and capability < MIN_COMPUTE_CAPABILITY:
            return VllmSupport("unsupported", None,
                               f"vLLM needs CUDA compute capability {MIN_COMPUTE_CAPABILITY}+, "
                               f"this GPU reports {capability}")
        if not (CUDA_PYTHON_RANGE[0] <= python_version <= CUDA_PYTHON_RANGE[1]):
            return VllmSupport("unsupported", None,
                               "vLLM's CUDA wheels need Python 3.10–3.13 and no matching "
                               "interpreter was found")
        return VllmSupport("supported", "cuda_wheel",
                           "Linux + NVIDIA CUDA is vLLM's primary platform, with prebuilt wheels",
                           requires_python=None)

    if rocm_ok:
        if rocm_version is not None and rocm_version < MIN_ROCM_VERSION:
            version_text = ".".join(str(part) for part in rocm_version)
            return VllmSupport("unsupported", None,
                               f"vLLM needs ROCm {MIN_ROCM_VERSION[0]}.{MIN_ROCM_VERSION[1]}+, "
                               f"this system reports {version_text}")
        return VllmSupport("supported", "rocm_wheel",
                           "Linux + AMD ROCm has prebuilt wheels, published for CPython 3.12 only",
                           requires_python=PINNED_PYTHON)

    if intel_gpu:
        return VllmSupport("unsupported", None,
                           "vLLM's Intel XPU backend publishes no wheels and needs a long "
                           "source build — out of scope for this setup script")

    return VllmSupport("unsupported", None,
                       "no supported GPU detected — vLLM's CPU backend is out of scope for "
                       "this benchmark, which measures accelerated inference")


def python_candidates(requires_python: tuple[int, int] | None,
                      current_version: tuple[int, int]) -> list[str]:
    """Interpreter names to try for the vLLM venv, best first."""
    if requires_python is not None:
        pinned = f"python{requires_python[0]}.{requires_python[1]}"
        return [pinned] if current_version != requires_python else [sys.executable, pinned]
    candidates = []
    if CUDA_PYTHON_RANGE[0] <= current_version <= CUDA_PYTHON_RANGE[1]:
        candidates.append(sys.executable)
    candidates += [f"python3.{minor}" for minor in range(13, 9, -1)]
    return candidates


def resolve_python(requires_python: tuple[int, int] | None,
                   current_version: tuple[int, int],
                   which_fn=shutil.which) -> str | None:
    for candidate in python_candidates(requires_python, current_version):
        if candidate == sys.executable or which_fn(candidate):
            return candidate
    return None


def vllm_install_command(method: str, python_exe: str, uv_available: bool) -> list[str]:
    """Argv that installs vLLM into the venv owned by `python_exe`."""
    extra = {
        "cuda_wheel": ["vllm", "--torch-backend=auto"] if uv_available else ["vllm"],
        "rocm_wheel": ["vllm", "--extra-index-url", ROCM_WHEEL_INDEX, "--upgrade"],
        "nightly_cu130": ["-U", "vllm", "--extra-index-url", NIGHTLY_CU130_INDEX],
    }[method]
    if uv_available:
        return ["uv", "pip", "install", "--python", python_exe] + extra
    return [python_exe, "-m", "pip", "install"] + extra


def find_vllm_binary(*, platform_name: str, venv_dir: Path = None,
                     which_fn=shutil.which, exists_fn=None) -> str | None:
    """Locate a `vllm` executable: this project's venv first, then vllm-metal's, then PATH."""
    venv_dir = venv_dir or config.VLLM_VENV
    exists_fn = exists_fn or (lambda path: Path(path).is_file())
    subdir = "Scripts" if platform_name == "Windows" else "bin"
    suffix = ".exe" if platform_name == "Windows" else ""
    candidates = [Path(venv_dir) / subdir / f"vllm{suffix}"]
    if platform_name == "Darwin":
        candidates.append(Path.home() / ".venv-vllm-metal" / "bin" / "vllm")
    for candidate in candidates:
        if exists_fn(candidate):
            return str(candidate)
    return which_fn("vllm")


def install_vllm(support: VllmSupport, *, log=print, run=subprocess.run,
                 venv_dir: Path = None) -> bool:  # pragma: no cover
    """Install vLLM per `support.method`. Real network/venv side effects — see AGENTS.md."""
    if not support.installable:
        return False
    if support.method == "metal_plugin":
        result = run(["bash", "-c", f"curl -fsSL {METAL_INSTALL_URL} | bash"])
        return result.returncode == 0

    venv_dir = Path(venv_dir or config.VLLM_VENV)
    python_exe = resolve_python(support.requires_python, sys.version_info[:2])
    if python_exe is None:
        pinned = support.requires_python
        log(f"No Python {pinned[0]}.{pinned[1]} interpreter found — vLLM's "
            f"{support.method} wheels require it")
        return False

    if not venv_dir.exists():
        log(f"Creating vLLM environment at {venv_dir} ...")
        created = run([python_exe, "-m", "venv", str(venv_dir)])
        if created.returncode != 0:
            return False
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    command = vllm_install_command(support.method, str(venv_python), bool(shutil.which("uv")))
    log(f"Installing vLLM ({support.method}) — this downloads several GB ...")
    return run(command).returncode == 0
