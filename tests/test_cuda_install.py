from types import SimpleNamespace

from scripts.setup.cuda_install import (
    CUDA_TOOLKIT_PACKAGE,
    cuda_toolkit_plan,
    run_cuda_toolkit_install,
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
