from types import SimpleNamespace

import pytest

from scripts.setup.intel_xpu_install import (
    ONEAPI_DNNL_PACKAGE,
    ONEAPI_SETVARS,
    ONEAPI_TOOLKIT_PACKAGE,
    ONEAPI_UNIFIED_VARS,
    intel_xpu_install_plan,
    oneapi_environment,
    oneapi_environment_script,
    parse_environment,
    run_intel_xpu_install,
    sycl_gpu_available,
)


def test_plan_installs_intel_compute_and_oneapi_build_dependencies():
    plan = intel_xpu_install_plan(
        'ID=ubuntu\nVERSION_ID="24.04"\n', user="tester",
    )
    flattened = [item for command in plan.commands for item in command]
    assert "ppa:kobuk-team/intel-graphics" in flattened
    assert "libze-intel-gpu1" in flattened
    assert "intel-ocloc" in flattened
    assert ONEAPI_TOOLKIT_PACKAGE == "intel-deep-learning-essentials-2026.1"
    assert ONEAPI_DNNL_PACKAGE == "intel-oneapi-dnnl-devel-2026.0"
    assert ONEAPI_TOOLKIT_PACKAGE in flattened
    assert ONEAPI_DNNL_PACKAGE in flattened
    assert "intel-oneapi-onedpl-devel" not in flattened
    assert plan.commands[-1] == ("usermod", "-aG", "render", "tester")
    assert any("linux-generic-hwe-24.04" in command for command in plan.commands)


def test_ubuntu_2604_plan_does_not_install_an_older_release_hwe_package():
    plan = intel_xpu_install_plan("ID=ubuntu\nVERSION_ID=26.04\n", user=None)
    assert not any("linux-generic-hwe-24.04" in command for command in plan.commands)


@pytest.mark.parametrize("release", [
    "ID=debian\nVERSION_ID=12\n",
    "ID=ubuntu\nVERSION_ID=22.04\n",
])
def test_plan_rejects_unsupported_linux_releases(release):
    with pytest.raises(ValueError, match="Ubuntu 24.04 or 26.04"):
        intel_xpu_install_plan(release, user="tester")


def test_plan_rejects_unsafe_user_name():
    with pytest.raises(ValueError, match="safely identify"):
        intel_xpu_install_plan("ID=ubuntu\nVERSION_ID=26.04\n", user="bad;name")


def test_install_uses_sudo_and_stops_on_failure():
    plan = intel_xpu_install_plan("ID=ubuntu\nVERSION_ID=26.04\n", user=None)
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    assert not run_intel_xpu_install(plan, run=run, geteuid=lambda: 1000, log=lambda _m: None)
    assert calls == [["sudo", *plan.commands[0]]]


def test_install_failure_identifies_command_and_exit_code():
    plan = intel_xpu_install_plan("ID=ubuntu\nVERSION_ID=26.04\n", user=None)
    messages = []

    assert not run_intel_xpu_install(
        plan,
        run=lambda _command: SimpleNamespace(returncode=100),
        geteuid=lambda: 1000,
        log=messages.append,
    )
    assert messages[-1] == "  Command failed with exit code 100: sudo apt-get update"


def test_install_omits_sudo_as_root():
    plan = intel_xpu_install_plan("ID=ubuntu\nVERSION_ID=24.04\n", user=None)
    calls = []

    def run(command):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    assert run_intel_xpu_install(plan, run=run, geteuid=lambda: 0, log=lambda _m: None)
    assert calls[0] == list(plan.commands[0])


def test_oneapi_environment_merges_sourced_values():
    def run(command, **kwargs):
        assert command == ["bash", "-c", f"source {ONEAPI_UNIFIED_VARS} >/dev/null && env"]
        assert kwargs["env"] == {"BASE": "yes"}
        return SimpleNamespace(returncode=0, stdout="PATH=/opt/intel/bin\nLD_LIBRARY_PATH=/opt/intel/lib\n")

    assert oneapi_environment(
        base_env={"BASE": "yes"}, run=run, is_file=lambda path: path == ONEAPI_UNIFIED_VARS,
    ) == {
        "BASE": "yes", "PATH": "/opt/intel/bin", "LD_LIBRARY_PATH": "/opt/intel/lib",
    }
    assert parse_environment("A=one=two\ninvalid\n") == {"A": "one=two"}


def test_oneapi_environment_prefers_pinned_unified_layout():
    assert oneapi_environment_script(is_file=lambda _path: True) == ONEAPI_UNIFIED_VARS


def test_oneapi_environment_falls_back_to_component_layout():
    assert oneapi_environment_script(
        is_file=lambda path: path == ONEAPI_SETVARS,
    ) == ONEAPI_SETVARS


def test_oneapi_environment_rejects_missing_initializers():
    assert oneapi_environment(base_env={}, is_file=lambda _path: False) is None


def test_oneapi_environment_rejects_failed_setvars():
    result = oneapi_environment(
        base_env={}, run=lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
        is_file=lambda path: path == ONEAPI_UNIFIED_VARS,
    )
    assert result is None


@pytest.mark.parametrize(("output", "expected"), [
    ("[level_zero:gpu] Intel(R) Arc(TM) Pro B65 Graphics", True),
    ("[opencl:cpu] Intel(R) Xeon", False),
    ("[cuda:gpu] NVIDIA GeForce", False),
])
def test_sycl_gpu_probe_requires_an_intel_gpu(output, expected):
    result = sycl_gpu_available(
        env={"ONEAPI": "yes"},
        run=lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=output, stderr="",
        ),
    )
    assert result is expected
