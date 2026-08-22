from types import SimpleNamespace

import pytest

from scripts.release.qualification_targets import qualification_target
from scripts.setup.cuda_install import (
    CUDA_TOOLKIT_PACKAGE,
    cuda_toolkit_plan,
    native_nvidia_driver_plan,
    nouveau_loaded,
    qualification_needs_native_nvidia_driver,
    run_cuda_toolkit_install,
    run_native_nvidia_driver_install,
)


def have(*names):
    return lambda name: name if name in names else None


def plan(**overrides):
    kwargs = {"is_wsl": True, "nvidia_ok": True, "nvcc_found": False,
              "which_fn": have("apt-get", "wget")}
    kwargs.update(overrides)
    return cuda_toolkit_plan(**kwargs)


def test_plan_installs_the_wsl_ubuntu_toolkit():
    commands = plan()
    assert commands[0][0] == "wget"
    assert "wsl-ubuntu" in commands[0][-1]
    assert commands[-1] == ["sudo", "apt-get", "install", "-y", CUDA_TOOLKIT_PACKAGE]


def test_plan_never_pulls_a_driver_bearing_metapackage():
    # These would install a Linux GPU driver and break WSL2's /dev/dxg passthrough.
    installed = plan()[-1]
    for forbidden in ("cuda", "cuda-12-x", "cuda-drivers"):
        assert forbidden not in installed[4:]


def test_plan_is_empty_unless_it_applies():
    assert plan(nvcc_found=True) == []
    assert plan(is_wsl=False) == []
    assert plan(nvidia_ok=False) == []
    assert plan(which_fn=have("wget")) == []


def test_plan_falls_back_to_curl_and_gives_up_without_a_downloader():
    assert plan(which_fn=have("apt-get", "curl"))[0][0] == "curl"
    assert plan(which_fn=have("apt-get")) == []


def test_install_reports_success_only_when_every_command_succeeds():
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    assert run_cuda_toolkit_install([["a"], ["b"]], log=lambda _m: None, run=run)
    assert calls == [["a"], ["b"]]


def test_install_stops_at_the_first_failing_command():
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    assert not run_cuda_toolkit_install([["a"], ["b"]], log=lambda _m: None, run=run)
    assert calls == [["a"]]


def test_install_reports_a_missing_binary_instead_of_raising():
    messages = []

    def run(_command):
        raise FileNotFoundError(2, "No such file or directory", "wget")

    assert not run_cuda_toolkit_install([["wget"]], log=messages.append, run=run)
    assert any("wget" in message for message in messages)


def test_native_nvidia_target_requires_driver_only_when_nvidia_smi_is_missing():
    target = qualification_target("nvidia-linux-llamacpp-cuda")
    assert qualification_needs_native_nvidia_driver(
        target, os_name="Linux", release="6.17.0-generic", nvidia_available=False,
    )
    assert not qualification_needs_native_nvidia_driver(
        target, os_name="Linux", release="6.17.0-generic", nvidia_available=True,
    )
    assert not qualification_needs_native_nvidia_driver(
        qualification_target("geforce-wsl2-llamacpp-cuda"), os_name="Linux",
        release="microsoft-standard-WSL2", nvidia_available=False,
    )
    assert not qualification_needs_native_nvidia_driver(
        qualification_target("dgx-spark-llamacpp-cuda"), os_name="Linux",
        release="6.14.0-nvidia", nvidia_available=False,
    )


def test_native_nvidia_target_rejects_wsl_before_installing_linux_driver():
    with pytest.raises(ValueError, match="requires native Linux"):
        qualification_needs_native_nvidia_driver(
            qualification_target("nvidia-linux-vllm-cuda"), os_name="Linux",
            release="microsoft-standard-WSL2", nvidia_available=False,
        )


def test_native_nvidia_plan_uses_ubuntu_hardware_selected_signed_driver():
    commands = native_nvidia_driver_plan(
        {"ID": "ubuntu", "VERSION_ID": "26.04"}, "7.0.0-30-generic",
    )
    assert commands == (
        ("apt-get", "update"),
        (
            "apt-get", "install", "-y", "ubuntu-drivers-common",
            "linux-headers-7.0.0-30-generic",
        ),
        ("ubuntu-drivers", "install"),
    )


def test_native_nvidia_plan_disables_loaded_nouveau_before_reboot():
    commands = native_nvidia_driver_plan(
        {"ID": "ubuntu", "VERSION_ID": "24.04"}, "6.8.0-90-generic",
        disable_nouveau=True,
    )
    assert commands[-2][0:2] == ("sh", "-c")
    assert "blacklist nouveau" in commands[-2][2]
    assert commands[-1] == ("update-initramfs", "-u")


def test_nouveau_detection_reads_the_loaded_kernel_module(tmp_path):
    module = tmp_path / "nouveau"
    assert not nouveau_loaded(module)
    module.mkdir()
    assert nouveau_loaded(module)


@pytest.mark.parametrize("release", [
    {"ID": "ubuntu", "VERSION_ID": "22.04"},
    {"ID": "debian", "VERSION_ID": "13"},
])
def test_native_nvidia_plan_rejects_unqualified_distributions(release):
    with pytest.raises(ValueError, match="requires Ubuntu 24.04 or 26.04"):
        native_nvidia_driver_plan(release, "6.17.0-generic")


def test_native_nvidia_install_is_privileged_and_stops_on_failure():
    commands = (("first",), ("second",))
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=1 if command[-1] == "second" else 0)

    with pytest.raises(RuntimeError, match="NVIDIA driver install command exited with 1"):
        run_native_nvidia_driver_install(commands, run=run)
    assert calls == [["sudo", "first"], ["sudo", "second"]]
