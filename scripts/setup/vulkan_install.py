"""Linux Vulkan build prerequisites for the managed llama.cpp runtime."""

import shutil
import subprocess
from pathlib import Path


VULKAN_HEADERS = (
    Path("/usr/include/vulkan/vulkan.h"),
    Path("/usr/local/include/vulkan/vulkan.h"),
)
SPIRV_HEADERS = (
    Path("/usr/include/spirv/unified1/spirv.h"),
    Path("/usr/local/include/spirv/unified1/spirv.h"),
)
PACKAGES = {
    "apt-get": (
        "git", "cmake", "build-essential", "glslc", "libvulkan-dev", "spirv-headers",
    ),
    "dnf": (
        "git", "cmake", "gcc-c++", "make", "glslc", "vulkan-loader-devel", "spirv-headers",
    ),
}


def missing_vulkan_build_requirements(*, which=shutil.which,
                                      is_file=lambda path: path.is_file()) -> tuple[str, ...]:
    missing = []
    for tool in ("git", "cmake"):
        if which(tool) is None:
            missing.append(tool)
    if not any(which(tool) for tool in ("c++", "g++", "clang++")):
        missing.append("C++ compiler")
    if which("glslc") is None:
        missing.append("glslc")
    if not any(is_file(path) for path in VULKAN_HEADERS):
        missing.append("Vulkan development headers")
    if not any(is_file(path) for path in SPIRV_HEADERS):
        missing.append("SPIR-V headers")
    return tuple(missing)


def vulkan_build_install_plan(missing: tuple[str, ...], *,
                              which=shutil.which) -> tuple[tuple[str, ...], ...] | None:
    if not missing:
        return ()
    manager = next((name for name in PACKAGES if which(name)), None)
    if manager is None:
        return None
    prefix = ("sudo", manager)
    update = (*prefix, "update") if manager == "apt-get" else None
    install = (*prefix, "install", "-y", *PACKAGES[manager])
    return (update, install) if update else (install,)


def run_vulkan_build_install(plan: tuple[tuple[str, ...], ...], *,
                             run=subprocess.run) -> bool:
    return all(run(list(command)).returncode == 0 for command in plan)
