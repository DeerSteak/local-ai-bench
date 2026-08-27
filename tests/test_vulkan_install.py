from types import SimpleNamespace

from scripts.setup.vulkan_install import (
    missing_vulkan_build_requirements, run_vulkan_build_install,
    vulkan_build_install_plan,
)


def test_requirement_probe_reports_each_missing_build_input():
    assert missing_vulkan_build_requirements(
        which=lambda _name: None, is_file=lambda _path: False,
    ) == ("glslc", "Vulkan development headers", "SPIR-V headers")


def test_requirement_probe_accepts_tools_and_headers_from_supported_prefixes():
    assert missing_vulkan_build_requirements(
        which=lambda name: f"/usr/bin/{name}", is_file=lambda _path: True,
    ) == ()


def test_apt_plan_installs_the_complete_upstream_build_toolchain():
    plan = vulkan_build_install_plan(
        ("glslc",), which=lambda name: "/usr/bin/apt-get" if name == "apt-get" else None,
    )
    assert plan == (
        ("sudo", "apt-get", "update"),
        ("sudo", "apt-get", "install", "-y", "glslc", "libvulkan-dev", "spirv-headers"),
    )


def test_dnf_plan_and_unknown_package_manager_are_explicit():
    missing = ("Vulkan development headers",)
    assert vulkan_build_install_plan(
        missing, which=lambda name: "/usr/bin/dnf" if name == "dnf" else None,
    ) == (("sudo", "dnf", "install", "-y", "glslc", "vulkan-loader-devel", "spirv-headers"),)
    assert vulkan_build_install_plan(missing, which=lambda _name: None) is None


def test_install_stops_at_the_first_failed_command():
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    assert not run_vulkan_build_install((("first",), ("second",)), run=run)
    assert calls == [["first"]]
