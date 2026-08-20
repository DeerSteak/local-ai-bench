from types import SimpleNamespace

import pytest

from scripts.release.qualification_targets import qualification_target
from scripts.setup.rocm_wsl_install import (
    qualification_needs_wsl_rocm, run_wsl_rocm_install, wsl_rocm_install_plan,
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


def test_wsl_rocm_install_downloads_and_runs_privileged_plan(tmp_path):
    plan = wsl_rocm_install_plan("ID=ubuntu\nVERSION_ID=24.04\n", temp_dir=tmp_path)
    downloads = []
    commands = []

    run_wsl_rocm_install(
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
        run_wsl_rocm_install(
            plan, download=lambda *_args: None, run=run, geteuid=lambda: 0,
        )
    assert commands == [list(plan.commands[0])]
