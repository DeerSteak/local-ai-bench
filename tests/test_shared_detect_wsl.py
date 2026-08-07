import pytest

from scripts.runtime.shared import Shared


@pytest.mark.parametrize("release", [
    "5.15.167.4-microsoft-standard-WSL2",
    "6.6.87.2-microsoft-standard-WSL2+",
    "4.4.0-19041-Microsoft",
])
def test_detect_wsl_recognizes_microsoft_kernels(release):
    assert Shared.detect_wsl("Linux", release) is True


@pytest.mark.parametrize("release", [
    "6.8.0-51-generic",
    "5.14.0-427.el9.x86_64",
    "",
])
def test_detect_wsl_rejects_native_linux_kernels(release):
    assert Shared.detect_wsl("Linux", release) is False


def test_detect_wsl_rejects_missing_release():
    assert Shared.detect_wsl("Linux", None) is False


@pytest.mark.parametrize("os_name", ["Windows", "Darwin", "FreeBSD"])
def test_detect_wsl_is_linux_only(os_name):
    # Native Windows names its releases numerically, but never claim WSL off-Linux.
    assert Shared.detect_wsl(os_name, "microsoft-standard-WSL2") is False
