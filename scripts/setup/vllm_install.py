"""vLLM support matrix, discovery, and installer — see docs/setup.md."""

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import sys

from scripts.runtime import config

METAL_INSTALL_URL = "https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh"
ROCM_WHEEL_INDEX = "https://wheels.vllm.ai/rocm/"
NIGHTLY_CU130_INDEX = "https://wheels.vllm.ai/nightly/cu130"
# Pointed at, never pulled — see docs/setup.md's Strix Halo note.
STRIX_HALO_TOOLBOX_URL = "https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes"

# vLLM's own floor for the CUDA wheels; below this the kernels aren't built.
MIN_COMPUTE_CAPABILITY = 7.5
MIN_ROCM_VERSION = (6, 3)

# gfx targets the prebuilt ROCm wheels ship kernels for; anything else is experimental.
VLLM_ROCM_WHEEL_TARGETS = ("gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1200", "gfx1201")

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
                          rocm_version: tuple[int, int] | None = None,
                          rocm_gfx_targets=None) -> VllmSupport:
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
        untargeted = [target for target in (rocm_gfx_targets or [])
                      if target not in VLLM_ROCM_WHEEL_TARGETS]
        if untargeted and not any(target in VLLM_ROCM_WHEEL_TARGETS
                                  for target in rocm_gfx_targets):
            return VllmSupport("experimental", "rocm_wheel",
                               f"vLLM's prebuilt ROCm wheels ship no kernels for "
                               f"{', '.join(untargeted)} — they target "
                               f"{', '.join(VLLM_ROCM_WHEEL_TARGETS)}. A TheRock-based "
                               f"container is the known-working route ({STRIX_HALO_TOOLBOX_URL}); "
                               "installing these wheels here may not produce a usable vLLM",
                               requires_python=PINNED_PYTHON)
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
    """Locate a `vllm`, system-first — matching the llama.cpp policy."""
    on_path = which_fn("vllm")
    if on_path:
        return on_path
    venv_dir = venv_dir or config.VLLM_VENV
    exists_fn = exists_fn or (lambda path: Path(path).is_file())
    subdir = "Scripts" if platform_name == "Windows" else "bin"
    suffix = ".exe" if platform_name == "Windows" else ""
    candidates = []
    if platform_name == "Darwin":
        candidates.append(Path.home() / ".venv-vllm-metal" / "bin" / "vllm")
    candidates.append(Path(venv_dir) / subdir / f"vllm{suffix}")
    for candidate in candidates:
        if exists_fn(candidate):
            return str(candidate)
    return None


# AMD's launcher bind-mounts this as the container's HF_HOME, so weights placed
# here are found by repo id — see docs/setup.md's Strix Halo note.
LAUNCHER_CACHE_HOMES = ("~/.local/share/vLLM/models",)
DEFAULT_HF_HOME = "~/.cache/huggingface"

LAUNCHER_NAMES = ("vllm-launch",)
LAUNCHER_CONF = Path("~/.local/share/vLLM/vllm-launch.conf")


def find_vllm_launcher(which_fn=shutil.which) -> str | None:
    """A platform wrapper around `vllm serve`, preferred over it — see docs/setup.md."""
    for name in LAUNCHER_NAMES:
        found = which_fn(name)
        if found:
            return found
    return None


def vllm_cache_home(launcher: str | None = None, env=None, exists_fn=None) -> Path:
    """HF cache vLLM actually reads. A platform launcher's own cache wins, because a
    containerised vLLM cannot see the host's default one."""
    env = os.environ if env is None else env
    exists_fn = exists_fn or (lambda path: path.is_dir())
    if launcher:
        for candidate in LAUNCHER_CACHE_HOMES:
            path = Path(candidate).expanduser()
            if exists_fn(path):
                return path
    return Path(env.get("HF_HOME") or DEFAULT_HF_HOME).expanduser()


def hf_cache_model_dir(cache_home: Path, repo: str) -> Path:
    """Where huggingface_hub stores `repo` inside a cache home."""
    return Path(cache_home) / "hub" / ("models--" + repo.replace("/", "--"))


def hf_cache_model_complete(cache_home: Path, repo: str) -> bool:
    """True once a snapshot of `repo` holds weights and the config beside them."""
    snapshots = hf_cache_model_dir(cache_home, repo) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any((snapshot / "config.json").is_file() and any(snapshot.glob("*.safetensors"))
               for snapshot in snapshots.iterdir() if snapshot.is_dir())


def parse_launcher_extra_args(text: str) -> list[str]:
    """`VLLM_EXTRA_ARGS=(...)`/`+=(...)` args a launcher conf injects into every run."""
    args = []
    for match in re.finditer(r"^\s*VLLM_EXTRA_ARGS\+?=\(([^)]*)\)", text or "", re.MULTILINE):
        args.extend(shlex.split(match.group(1)))
    return args


def read_launcher_extra_args(path: Path = None) -> list[str]:
    path = Path(path or LAUNCHER_CONF).expanduser()
    try:
        return parse_launcher_extra_args(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def vllm_server_reachable(url: str = None, timeout: float = 2.0, open_fn=None) -> bool:
    """True if an OpenAI-compatible vLLM server answers at `url`."""
    url = url or config.VLLM_URL
    if open_fn is None:  # pragma: no cover — real socket
        import urllib.request
        open_fn = urllib.request.urlopen
    try:
        with open_fn(f"{url}/v1/models", timeout=timeout) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except Exception:
        return False


def find_vllm_server(ports=None, timeout: float = 2.0, open_fn=None) -> str | None:
    """URL of an already-running vLLM, or None."""
    for port in (ports if ports is not None else config.VLLM_DISCOVERY_PORTS):
        url = f"http://localhost:{port}"
        if vllm_server_reachable(url, timeout=timeout, open_fn=open_fn):
            return url
    return None


def install_vllm(support: VllmSupport, *, log=print, run=subprocess.run,
                 venv_dir: Path = None) -> bool:  # pragma: no cover
    """Install vLLM per `support.method`. Real network/venv side effects."""
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
