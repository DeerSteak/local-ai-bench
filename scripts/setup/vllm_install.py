"""vLLM support matrix, discovery, and installer — see docs/setup.md."""

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import sys
import json
import urllib.request

from packaging.version import InvalidVersion, Version

from scripts.runtime import config
from scripts.runtime.log_redaction import redact_log_text

ROCM_WHEEL_INDEX = "https://wheels.vllm.ai/rocm/"
DGX_CU130_VERSION = "0.27.1"
DGX_CU130_INDEX = f"https://wheels.vllm.ai/{DGX_CU130_VERSION}/cu130"
# vLLM's own floor for the CUDA wheels; below this the kernels aren't built.
MIN_COMPUTE_CAPABILITY = 7.5
MIN_ROCM_VERSION = (6, 3)

# gfx targets the prebuilt ROCm wheels ship kernels for; anything else is experimental.
VLLM_ROCM_WHEEL_TARGETS = (
    "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
)

# The ROCm and Metal builds publish CPython 3.12 wheels only; CUDA spans a range.
CUDA_PYTHON_RANGE = ((3, 10), (3, 13))
PINNED_PYTHON = (3, 12)


@dataclass(frozen=True)
class VllmSupport:
    status: str            # "supported" | "experimental" | "unsupported"
    method: str | None     # "cuda_wheel" | "rocm_wheel" | "cu130_wheel"
    reason: str
    requires_python: tuple[int, int] | None = None
    # Set only when the sole obstacle is the interpreter, so setup knows an offer can clear it.
    needs_python_bootstrap: bool = False

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
                          rocm_gfx_targets=None,
                          which_fn=shutil.which) -> VllmSupport:
    """Whether setup can install vLLM here, and how. See docs/setup.md's support table."""
    if os_name == "Windows":
        return VllmSupport("unsupported", None,
                           "vLLM has no upstream Windows support — install WSL2 and run "
                           "setup inside it, where the Linux CUDA path applies. "
                           "See docs/setup.md#vllm-on-windows-via-wsl2")

    if os_name == "Darwin":
        return VllmSupport("unsupported", None,
                           "vLLM on macOS is out of scope for this project — its catalog's "
                           "AWQ/GPTQ weights aren't loadable by the community vllm-metal "
                           "plugin's MLX backend")

    if os_name != "Linux":
        return VllmSupport("unsupported", None, f"vLLM has no build for {os_name}")

    if nvidia_ok:
        if is_dgx_spark(machine, gpu_names):
            return VllmSupport("experimental", "cu130_wheel",
                               "DGX Spark (GB10, sm_121) is not covered by stock wheels — "
                               "the reviewed CUDA 13 build is the only working path, and "
                               "plain wheels would silently install CPU-only PyTorch",
                               requires_python=PINNED_PYTHON)
        capability = parse_compute_capability(compute_cap)
        if capability is not None and capability < MIN_COMPUTE_CAPABILITY:
            return VllmSupport("unsupported", None,
                               f"vLLM needs CUDA compute capability {MIN_COMPUTE_CAPABILITY}+, "
                               f"this GPU reports {capability}")
        # vLLM gets its own venv, so an out-of-range interpreter here is fine as long as
        # an in-range one exists on PATH for that venv to be built from.
        if (not (CUDA_PYTHON_RANGE[0] <= python_version <= CUDA_PYTHON_RANGE[1])
                and resolve_python(None, python_version, which_fn) is None):
            return VllmSupport("unsupported", None,
                               "vLLM's CUDA wheels need Python 3.10–3.13 and no matching "
                               "interpreter was found",
                               needs_python_bootstrap=True)
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
                                  for target in (rocm_gfx_targets or [])):
            return VllmSupport("experimental", "rocm_wheel",
                               f"vLLM's prebuilt ROCm wheels ship no kernels for "
                               f"{', '.join(untargeted)} — they target "
                               f"{', '.join(VLLM_ROCM_WHEEL_TARGETS)}; installing these wheels "
                               "here may not produce a usable vLLM",
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


UV_INSTALLER_URL = "https://astral.sh/uv/install.sh"


def python_bootstrap_plan(*, python_version: tuple[int, int],
                          requires_python: tuple[int, int] | None = None,
                          which_fn=shutil.which) -> list[list[str]]:
    """Commands that put a vLLM-compatible interpreter on PATH, empty when one already is.
    Needed on distros whose only system Python is newer than vLLM's wheels — see docs/setup.md."""
    if resolve_python(requires_python, python_version, which_fn) is not None:
        return []
    plan = []
    if which_fn("uv") is None:
        plan.append(["sh", "-c", f"curl -LsSf {UV_INSTALLER_URL} | sh"])
    plan.append(["uv", "python", "install", f"{PINNED_PYTHON[0]}.{PINNED_PYTHON[1]}"])
    return plan


def run_python_bootstrap(plan: list[list[str]], *, log=print,
                         run=subprocess.run) -> bool:
    """Execute a `python_bootstrap_plan`, returning whether every command succeeded."""
    # uv installs its shims here, so PATH has to admit them before the next command runs.
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = local_bin + os.pathsep + os.environ.get("PATH", "")
    for command in plan:
        log(f"  Running: {shlex.join(command)}")
        try:
            failed = run(command).returncode != 0
        except OSError as exc:
            log(f"  Could not run {command[0]}: {exc}")
            failed = True
        if failed:
            log("  Bootstrap failed — continuing without vLLM")
            return False
    return True


# `vllm bench` deps are not in the base package — see docs/workloads.md#vllm-bench.
VLLM_PACKAGE = "vllm[bench]"
VLLM_PYPI_URL = "https://pypi.org/pypi/vllm/json"


def normalize_vllm_version(value) -> str:
    text = str(value or "").strip()
    try:
        version = Version(text)
    except InvalidVersion as exc:
        raise ValueError("Enter a vLLM version such as 0.10.2.") from exc
    if version.is_prerelease or version.is_devrelease:
        raise ValueError("Stable vLLM version selection does not accept prereleases or nightlies.")
    return str(version)


def normalize_exact_vllm_version(value) -> str:
    """Accept a fully pinned stable or wheel-channel development build."""
    text = str(value or "").strip()
    try:
        return str(Version(text))
    except InvalidVersion as exc:
        raise ValueError("Enter an exact vLLM wheel version.") from exc


def fetch_vllm_versions(*, opener=urllib.request.urlopen) -> list[str]:
    with opener(VLLM_PYPI_URL, timeout=15) as response:
        payload = json.load(response)
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, dict):
        raise ValueError("PyPI returned invalid vLLM release metadata")
    versions = []
    for raw, files in releases.items():
        try:
            normalized = normalize_vllm_version(raw)
        except ValueError:
            continue
        if isinstance(files, list) and any(
                isinstance(item, dict) and not item.get("yanked") for item in files):
            versions.append(Version(normalized))
    return [str(version) for version in sorted(set(versions), reverse=True)[:10]]


def vllm_install_command(method: str, python_exe: str, uv_available: bool,
                         version: str | None = None,
                         index_url: str | None = None) -> list[str]:
    """Argv that installs vLLM into the venv owned by `python_exe`."""
    normalized = normalize_exact_vllm_version(version) if version else None
    package = f"vllm[bench]=={normalized}" if normalized else VLLM_PACKAGE
    extra = {
        "cuda_wheel": ([package, "--torch-backend=auto"] if uv_available else [package]),
        "rocm_wheel": [package, "--extra-index-url", ROCM_WHEEL_INDEX, "--upgrade"],
        "cu130_wheel": [
            "-U", f"vllm[bench]=={normalized or DGX_CU130_VERSION}",
            "--extra-index-url", index_url or DGX_CU130_INDEX,
        ],
    }[method]
    if uv_available:
        return ["uv", "pip", "install", "--python", python_exe] + extra
    return [python_exe, "-m", "pip", "install"] + extra


def find_vllm_binary(*, platform_name: str, venv_dir: Path | None = None,
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


def hf_cache_snapshot_dir(cache_home: Path, repo: str) -> Path | None:
    """Snapshot vLLM resolves for the repo's current main ref, with a legacy fallback."""
    model_dir = hf_cache_model_dir(cache_home, repo)
    snapshots = model_dir / "snapshots"
    try:
        revision = (model_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision:
        snapshot = snapshots / revision
        return snapshot if snapshot.is_dir() else None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()] \
        if snapshots.is_dir() else []
    complete = [path for path in candidates
                if (path / "config.json").is_file() and any(path.glob("*.safetensors"))]
    configured = [path for path in candidates if (path / "config.json").is_file()]
    usable = complete or configured
    return max(usable, key=lambda path: path.stat().st_mtime_ns) if usable else None


def hf_cache_model_complete(cache_home: Path, repo: str) -> bool:
    """True once a snapshot of `repo` holds weights and the config beside them."""
    snapshot = hf_cache_snapshot_dir(cache_home, repo)
    return bool(snapshot and (snapshot / "config.json").is_file()
                and any(snapshot.glob("*.safetensors")))


def parse_launcher_extra_args(text: str | None) -> list[str]:
    """`VLLM_EXTRA_ARGS=(...)`/`+=(...)` args a launcher conf injects into every run."""
    args = []
    for match in re.finditer(r"^\s*VLLM_EXTRA_ARGS\+?=\(([^)]*)\)", text or "", re.MULTILINE):
        args.extend(shlex.split(match.group(1)))
    return args


def read_launcher_extra_args(path: Path | None = None) -> list[str]:
    path = Path(path or LAUNCHER_CONF).expanduser()
    try:
        return parse_launcher_extra_args(path.read_text(encoding="utf-8"))
    except OSError:
        return []


SENSITIVE_LAUNCHER_FLAGS = {"--api-key", "--token", "--hf-token", "--password"}


def redact_launcher_extra_args(args: list[str]) -> list[str]:
    """Sanitize launcher flags before setup logs or persists them."""
    redacted = []
    hide_next = False
    for arg in args:
        if hide_next:
            redacted.append("<secret>")
            hide_next = False
            continue
        flag = arg.split("=", 1)[0].lower()
        if flag in SENSITIVE_LAUNCHER_FLAGS:
            if "=" in arg:
                redacted.append(f"{arg.split('=', 1)[0]}=<secret>")
            else:
                redacted.append(arg)
                hide_next = True
            continue
        redacted.append(redact_log_text(arg))
    return redacted


def missing_python_headers(include_dir: str | None, exists_fn=None) -> str | None:
    """Path of the absent Python.h, or None. Triton JIT-compiles a CUDA helper at
    import time, so vLLM cannot start without the development headers."""
    if not include_dir:
        return None
    exists_fn = exists_fn or (lambda path: Path(path).is_file())
    header = Path(include_dir) / "Python.h"
    return None if exists_fn(header) else str(header)


def python_include_dir(python_exe: str, run=subprocess.run) -> str | None:  # pragma: no cover — subprocess
    """The include directory of another interpreter, which may not be this one."""
    try:
        result = run([python_exe, "-c", "import sysconfig; print(sysconfig.get_paths()['include'])"],
                     capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def python_version_from_include_dir(include_dir: str | None) -> tuple[int, int] | None:
    """Version whose headers are missing, read from its include path. The vLLM venv's
    interpreter is often not the one running setup."""
    match = re.search(r"python(\d+)\.(\d+)", str(include_dir or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def running_as_root() -> bool:
    """Windows has no geteuid, and the package managers this gates are POSIX-only —
    treating its absence as "not root" keeps the sudo prefix correct everywhere."""
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def python_dev_package_command(package_manager: str, python_version: tuple[int, int],
                               which_fn=shutil.which) -> list[str] | None:
    """Command installing the Python headers Triton needs, for a known package manager."""
    major, minor = python_version
    if package_manager == "apt-get":
        package = f"python{major}.{minor}-dev"
    elif package_manager == "dnf":
        package = f"python{major}-devel"
    elif package_manager == "zypper":
        package = f"python{major}{minor}-devel"
    else:
        return None
    if not which_fn(package_manager):
        return None
    prefix = [] if running_as_root() else ["sudo"]
    yes = ["-y"] if package_manager != "zypper" else ["--non-interactive"]
    return prefix + [package_manager, "install", *yes, package]


# FlashInfer and Triton JIT-compile kernels at runtime; ninja drives those builds.
VLLM_BUILD_TOOLS = ("ninja",)


def missing_build_tools(venv_dir: Path, exists_fn=None) -> list[str]:
    """Build tools absent from the vLLM venv. Installed with pip, so no sudo is needed."""
    exists_fn = exists_fn or (lambda path: Path(path).is_file())
    bin_dir = Path(venv_dir) / ("Scripts" if os.name == "nt" else "bin")
    return [tool for tool in VLLM_BUILD_TOOLS if not exists_fn(bin_dir / tool)]


def build_tools_command(python_exe: str, tools) -> list[str] | None:
    return [python_exe, "-m", "pip", "install", *tools] if tools else None


def install_vllm_build_tools(venv_dir: Path, *, log=print,
                             run=subprocess.run) -> bool:
    missing = missing_build_tools(venv_dir)
    if not missing:
        return True
    python = Path(venv_dir) / ("Scripts" if os.name == "nt" else "bin") / \
        ("python.exe" if os.name == "nt" else "python")
    command = build_tools_command(str(python), missing)
    log(f"Installing vLLM build tools ({', '.join(missing)}) ...")
    return command is not None and run(command).returncode == 0


def vllm_server_reachable(url: str | None = None, timeout: float = 2.0, open_fn=None) -> bool:
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
                 venv_dir: Path | None = None,
                 version: str | None = None,
                 index_url: str | None = None) -> bool:  # pragma: no cover
    """Install vLLM per `support.method`. Real network/venv side effects."""
    if support.method is None:
        return False

    venv_dir = Path(venv_dir or config.VLLM_VENV)
    python_exe = resolve_python(support.requires_python, sys.version_info[:2])
    if python_exe is None:
        pinned = support.requires_python
        wanted = f"Python {pinned[0]}.{pinned[1]}" if pinned else "a suitable Python"
        log(f"No {wanted} interpreter found — vLLM's {support.method} wheels require it")
        return False

    if not venv_dir.exists():
        log(f"Creating vLLM environment at {venv_dir} ...")
        created = run([python_exe, "-m", "venv", str(venv_dir)])
        if created.returncode != 0:
            return False
    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
    command = vllm_install_command(
        support.method, str(venv_python), bool(shutil.which("uv")), version, index_url,
    )
    log(f"Installing vLLM ({support.method}) — this downloads several GB ...")
    return run(command).returncode == 0 and install_vllm_build_tools(
        venv_dir, log=log, run=run,
    )
