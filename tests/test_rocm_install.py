from types import SimpleNamespace

import pytest

from scripts.release.qualification_targets import qualification_target
from scripts.setup.rocm_install import (
    native_rocm_install_plan, qualification_needs_native_rocm,
    qualification_needs_wsl_rocm, run_rocm_install, wsl_rocm_install_plan,
)


def test_wsl_rocm_plan_pins_ubuntu_2404_installer(tmp_path):
    plan = wsl_rocm_install_plan('ID=ubuntu\nVERSION_ID="24.04"\n', temp_dir=tmp_path)
    assert plan.package_url == (
        "https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/"
        "amdgpu-install_7.2.70200-1_all.deb"
    )
    assert plan.commands == (
        ("apt-get", "update"),
        ("apt-get", "install", "-y", str(plan.package_path)),
        ("amdgpu-install", "-y", "--usecase=wsl,rocm", "--no-dkms"),
    )


def test_wsl_rocm_plan_supports_ubuntu_2204(tmp_path):
    plan = wsl_rocm_install_plan("ID=ubuntu\nVERSION_ID=22.04\n", temp_dir=tmp_path)
    assert "/jammy/" in plan.package_url


def test_native_rocm_plan_pins_supported_ubuntu_kernel_and_permissions(tmp_path):
    plan = native_rocm_install_plan(
        'ID=ubuntu\nVERSION_ID="24.04"\n', "6.17.0-14-generic",
        user="ben", temp_dir=tmp_path,
    )
    assert plan.package_url == (
        "https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/"
        "amdgpu-install_7.2.1.70201-1_all.deb"
    )
    assert plan.commands == (
        ("apt-get", "install", "-y", str(plan.package_path)),
        ("apt-get", "update"),
        ("amdgpu-install", "-y", "--usecase=graphics,rocm"),
        ("usermod", "-aG", "render,video", "ben"),
    )


@pytest.mark.parametrize("kernel", ["7.0.0-30-generic", "6.14.0-1-generic", "unknown"])
def test_native_rocm_plan_rejects_unsupported_kernel(kernel):
    with pytest.raises(ValueError, match="requires kernel 6.8 or 6.17"):
        native_rocm_install_plan(
            'ID=ubuntu\nVERSION_ID="24.04"\n', kernel, user="ben",
        )


@pytest.mark.parametrize("release", [
    "ID=ubuntu\nVERSION_ID=26.04\n",
    "ID=debian\nVERSION_ID=13\n",
])
def test_wsl_rocm_plan_rejects_unsupported_distributions(release):
    with pytest.raises(ValueError, match="supports Ubuntu 22.04 or 24.04"):
        wsl_rocm_install_plan(release)


def test_rocm_wsl_target_requires_install_only_when_runtime_is_missing():
    target = qualification_target("radeon-wsl2-llamacpp-rocm")
    assert qualification_needs_wsl_rocm(
        target, os_name="Linux", release="microsoft-standard-WSL2", rocm_available=False,
    )
    assert not qualification_needs_wsl_rocm(
        target, os_name="Linux", release="microsoft-standard-WSL2", rocm_available=True,
    )


def test_rocm_wsl_target_rejects_native_linux_before_install():
    target = qualification_target("radeon-wsl2-llamacpp-rocm")
    with pytest.raises(ValueError, match="requires WSL2"):
        qualification_needs_wsl_rocm(
            target, os_name="Linux", release="6.14-generic", rocm_available=False,
        )


def test_non_wsl_target_does_not_request_wsl_rocm_install():
    target = qualification_target("radeon-linux-llamacpp-rocm")
    assert not qualification_needs_wsl_rocm(
        target, os_name="Linux", release="6.14-generic", rocm_available=False,
    )


def test_native_rocm_target_requires_install_only_when_runtime_is_missing():
    target = qualification_target("radeon-linux-llamacpp-rocm")
    assert qualification_needs_native_rocm(
        target, os_name="Linux", release="6.17.0-generic", rocm_available=False,
    )
    assert not qualification_needs_native_rocm(
        target, os_name="Linux", release="6.17.0-generic", rocm_available=True,
    )
    wsl_target = qualification_target("radeon-wsl2-llamacpp-rocm")
    assert not qualification_needs_native_rocm(
        wsl_target, os_name="Linux", release="microsoft-standard-WSL2",
        rocm_available=False,
    )


def test_wsl_rocm_install_downloads_and_runs_privileged_plan(tmp_path):
    plan = wsl_rocm_install_plan("ID=ubuntu\nVERSION_ID=24.04\n", temp_dir=tmp_path)
    downloads = []
    commands = []

    run_rocm_install(
        plan,
        download=lambda url, path: downloads.append((url, path)),
        run=lambda command: commands.append(command) or SimpleNamespace(returncode=0),
        geteuid=lambda: 1000,
    )

    assert downloads == [(plan.package_url, plan.package_path)]
    assert commands == [["sudo", *command] for command in plan.commands]


def test_wsl_rocm_install_stops_at_first_failed_command(tmp_path):
    plan = wsl_rocm_install_plan("ID=ubuntu\nVERSION_ID=24.04\n", temp_dir=tmp_path)
    commands = []

    def run(command):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    with pytest.raises(RuntimeError, match="ROCm install command exited with 1"):
        run_rocm_install(
            plan, download=lambda *_args: None, run=run, geteuid=lambda: 0,
        )
    assert commands == [list(plan.commands[0])]
