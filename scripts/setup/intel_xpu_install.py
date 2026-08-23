"""Intel GPU compute and oneAPI prerequisites for native Linux setup."""

from dataclasses import dataclass
import os
import re
import shlex
import subprocess
from pathlib import Path

from scripts.setup.rocm_install import parse_os_release


SUPPORTED_UBUNTU = {"24.04", "26.04"}
ONEAPI_SETVARS = "/opt/intel/oneapi/setvars.sh"
ONEAPI_UNIFIED_VARS = "/opt/intel/oneapi/2026.1/oneapi-vars.sh"
ONEAPI_TOOLKIT_PACKAGE = "intel-deep-learning-essentials-2026.1"
ONEAPI_DNNL_PACKAGE = "intel-oneapi-dnnl-devel-2026.0"
ONEAPI_KEY_URL = (
    "https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB"
)


@dataclass(frozen=True)
class IntelXpuInstallPlan:
    commands: tuple[tuple[str, ...], ...]


def intel_xpu_install_plan(os_release: str, *, user: str | None) -> IntelXpuInstallPlan:
    release = parse_os_release(os_release)
    distribution = release.get("ID", "").lower()
    version = release.get("VERSION_ID", "")
    if distribution != "ubuntu" or version not in SUPPORTED_UBUNTU:
        detected = f"{distribution or 'unknown'} {version or 'unknown'}"
        raise ValueError(
            "Intel Arc XPU setup supports Ubuntu 24.04 or 26.04; "
            f"detected {detected}"
        )
    if user and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user):
        raise ValueError("could not safely identify the user for Intel GPU permissions")
    commands = [
        ("apt-get", "update"),
        ("apt-get", "install", "-y", "software-properties-common", "gpg-agent", "wget"),
        ("add-apt-repository", "-y", "ppa:kobuk-team/intel-graphics"),
        ("apt-get", "update"),
        (
            "apt-get", "install", "-y", "libze-intel-gpu1", "libze1",
            "intel-metrics-discovery", "intel-opencl-icd", "clinfo", "intel-gsc",
            "libze-dev", "intel-ocloc", "xpu-smi",
        ),
        (
            "bash", "-c",
            f"wget -qO- {ONEAPI_KEY_URL} | gpg --dearmor --yes "
            "--output /usr/share/keyrings/oneapi-archive-keyring.gpg",
        ),
        (
            "bash", "-c",
            "echo 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] "
            "https://apt.repos.intel.com/oneapi all main' "
            "> /etc/apt/sources.list.d/oneAPI.list",
        ),
        ("apt-get", "update"),
        ("apt-get", "install", "-y", ONEAPI_TOOLKIT_PACKAGE, ONEAPI_DNNL_PACKAGE),
    ]
    if version == "24.04":
        commands.insert(2, ("apt-get", "install", "-y", "linux-generic-hwe-24.04"))
    if user:
        commands.append(("usermod", "-aG", "render", user))
    return IntelXpuInstallPlan(tuple(commands))


def run_intel_xpu_install(plan: IntelXpuInstallPlan, *, log=print,
                          run=subprocess.run, geteuid=None) -> bool:
    effective_uid = geteuid or getattr(os, "geteuid", lambda: 1)
    prefix = [] if effective_uid() == 0 else ["sudo"]
    for command in plan.commands:
        argv = [*prefix, *command]
        log(f"  Running: {' '.join(argv)}")
        try:
            result = run(argv)
        except OSError as exc:
            log(f"  Could not run {command[0]}: {exc}")
            return False
        if result.returncode != 0:
            log(f"  Command failed with exit code {result.returncode}: {' '.join(argv)}")
            return False
    return True


def parse_environment(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def oneapi_environment_script(*, is_file=None) -> str | None:
    is_file = is_file or (lambda value: Path(value).is_file())
    return next(
        (script for script in (ONEAPI_UNIFIED_VARS, ONEAPI_SETVARS) if is_file(script)),
        None,
    )


def oneapi_environment(*, base_env=None, run=subprocess.run, is_file=None) -> dict[str, str] | None:
    base = dict(os.environ if base_env is None else base_env)
    script = oneapi_environment_script(is_file=is_file)
    if script is None:
        return None
    try:
        result = run(
            ["bash", "-c", f"source {shlex.quote(script)} >/dev/null && env"],
            capture_output=True, text=True, env=base,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {**base, **parse_environment(result.stdout)}


def sycl_gpu_available(*, env=None, run=subprocess.run) -> bool:
    try:
        result = run(
            ["sycl-ls"], capture_output=True, text=True, timeout=30, env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    return result.returncode == 0 and "gpu" in output and "intel" in output
