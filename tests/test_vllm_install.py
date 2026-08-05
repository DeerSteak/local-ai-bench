import sys
from pathlib import Path

from scripts.setup.vllm_install import (
    ROCM_WHEEL_INDEX,
    NIGHTLY_CU130_INDEX,
    find_vllm_binary,
    is_dgx_spark,
    parse_compute_capability,
    python_candidates,
    resolve_python,
    vllm_install_command,
    vllm_platform_support,
)


def support(**overrides):
    kwargs = {"os_name": "Linux", "machine": "x86_64", "python_version": (3, 12)}
    kwargs.update(overrides)
    return vllm_platform_support(**kwargs)


# ── platform matrix ──

def test_linux_nvidia_is_supported_via_cuda_wheel():
    result = support(nvidia_ok=True, compute_cap="8.9")
    assert (result.status, result.method) == ("supported", "cuda_wheel")
    assert result.requires_python is None


def test_linux_nvidia_accepts_full_python_range():
    for minor in (10, 11, 12, 13):
        assert support(nvidia_ok=True, python_version=(3, minor)).method == "cuda_wheel"


def test_python_outside_cuda_range_is_unsupported():
    for version in ((3, 9), (3, 14)):
        result = support(nvidia_ok=True, python_version=version)
        assert result.status == "unsupported"
        assert "3.10" in result.reason


def test_old_compute_capability_is_unsupported():
    result = support(nvidia_ok=True, compute_cap="7.0")
    assert result.status == "unsupported"
    assert "7.5" in result.reason


def test_compute_capability_exactly_at_the_floor_is_supported():
    assert support(nvidia_ok=True, compute_cap="7.5").method == "cuda_wheel"


def test_unknown_compute_capability_does_not_block_install():
    assert support(nvidia_ok=True, compute_cap=None).method == "cuda_wheel"
    assert support(nvidia_ok=True, compute_cap="N/A").method == "cuda_wheel"


def test_dgx_spark_uses_the_cuda_13_nightly_path():
    result = support(nvidia_ok=True, machine="aarch64", gpu_names=["NVIDIA GB10"])
    assert (result.status, result.method) == ("experimental", "nightly_cu130")
    assert result.requires_python == (3, 12)


def test_aarch64_nvidia_that_is_not_a_dgx_spark_takes_the_normal_cuda_path():
    result = support(nvidia_ok=True, machine="aarch64", gpu_names=["NVIDIA GH200"])
    assert (result.status, result.method) == ("supported", "cuda_wheel")


def test_gb10_on_x86_is_not_treated_as_a_dgx_spark():
    assert support(nvidia_ok=True, gpu_names=["GB10"]).method == "cuda_wheel"


def test_linux_rocm_is_supported_and_pins_python_312():
    result = support(rocm_ok=True, rocm_version=(6, 4))
    assert (result.status, result.method) == ("supported", "rocm_wheel")
    assert result.requires_python == (3, 12)


def test_rocm_below_the_minimum_is_unsupported():
    result = support(rocm_ok=True, rocm_version=(6, 2))
    assert result.status == "unsupported"
    assert "6.2" in result.reason


def test_rocm_exactly_at_the_minimum_is_supported():
    assert support(rocm_ok=True, rocm_version=(6, 3)).method == "rocm_wheel"


def test_unknown_rocm_version_does_not_block_install():
    assert support(rocm_ok=True, rocm_version=None).method == "rocm_wheel"


def test_nvidia_wins_over_rocm_when_both_are_reported():
    assert support(nvidia_ok=True, rocm_ok=True, rocm_version=(6, 0)).method == "cuda_wheel"


def test_apple_silicon_uses_the_metal_plugin():
    result = support(os_name="Darwin", machine="arm64")
    assert (result.status, result.method) == ("experimental", "metal_plugin")


def test_intel_mac_is_unsupported():
    result = support(os_name="Darwin", machine="x86_64")
    assert result.status == "unsupported"
    assert "Apple Silicon" in result.reason


def test_windows_is_unsupported_even_with_an_nvidia_gpu():
    result = support(os_name="Windows", nvidia_ok=True, compute_cap="8.9")
    assert result.status == "unsupported"
    assert "WSL2" in result.reason


def test_intel_xpu_is_unsupported():
    result = support(intel_gpu=True)
    assert result.status == "unsupported"
    assert "XPU" in result.reason


def test_cpu_only_linux_is_unsupported():
    result = support()
    assert (result.status, result.method) == ("unsupported", None)
    assert result.installable is False


def test_unknown_os_is_unsupported():
    assert support(os_name="FreeBSD", nvidia_ok=True).status == "unsupported"


# ── helpers ──

def test_parse_compute_capability_tolerates_junk():
    assert parse_compute_capability(" 8.9 ") == 8.9
    assert parse_compute_capability(None) is None
    assert parse_compute_capability("[N/A]") is None


def test_is_dgx_spark_matches_case_insensitively_on_arm():
    assert is_dgx_spark("arm64", ["nvidia gb10 superchip"])
    assert not is_dgx_spark("aarch64", [])
    assert not is_dgx_spark("aarch64", None)


def test_python_candidates_pins_the_required_minor():
    assert python_candidates((3, 12), (3, 11)) == ["python3.12"]
    assert python_candidates((3, 12), (3, 12))[0] == sys.executable


def test_python_candidates_prefers_current_interpreter_when_in_range():
    assert python_candidates(None, (3, 11))[0] == sys.executable
    assert python_candidates(None, (3, 9))[0] != sys.executable
    assert "python3.13" in python_candidates(None, (3, 9))


def test_resolve_python_returns_none_when_the_pinned_version_is_missing():
    assert resolve_python((3, 12), (3, 11), which_fn=lambda _: None) is None
    assert resolve_python((3, 12), (3, 11), which_fn=lambda name: f"/usr/bin/{name}") == "python3.12"


def test_install_commands_target_the_venv_interpreter():
    uv = vllm_install_command("cuda_wheel", "/v/bin/python", uv_available=True)
    assert uv[:5] == ["uv", "pip", "install", "--python", "/v/bin/python"]
    assert "--torch-backend=auto" in uv

    pip = vllm_install_command("cuda_wheel", "/v/bin/python", uv_available=False)
    assert pip == ["/v/bin/python", "-m", "pip", "install", "vllm"]
    assert "--torch-backend=auto" not in pip  # a pip-only flag would fail the install


def test_rocm_and_nightly_commands_use_their_own_indexes():
    rocm = vllm_install_command("rocm_wheel", "/v/bin/python", uv_available=False)
    assert ROCM_WHEEL_INDEX in rocm
    nightly = vllm_install_command("nightly_cu130", "/v/bin/python", uv_available=True)
    assert NIGHTLY_CU130_INDEX in nightly


def test_find_vllm_binary_prefers_a_system_install():
    assert find_vllm_binary(
        platform_name="Linux", venv_dir=Path("/proj/vllm-env"),
        exists_fn=lambda _: True, which_fn=lambda _: "/usr/bin/vllm",
    ) == "/usr/bin/vllm"


def test_find_vllm_binary_falls_back_to_the_project_venv():
    venv = Path("/proj/vllm-env")
    assert find_vllm_binary(
        platform_name="Linux", venv_dir=venv,
        exists_fn=lambda path: str(path).endswith("vllm-env/bin/vllm"),
        which_fn=lambda _: None,
    ) == str(venv / "bin" / "vllm")


def test_find_vllm_binary_prefers_the_metal_venv_over_the_project_venv_on_macos():
    metal = str(Path.home() / ".venv-vllm-metal" / "bin" / "vllm")
    assert find_vllm_binary(
        platform_name="Darwin", venv_dir=Path("/proj/vllm-env"),
        exists_fn=lambda _: True, which_fn=lambda _: None,
    ) == metal


def test_find_vllm_binary_returns_none_when_nothing_is_installed():
    assert find_vllm_binary(platform_name="Linux", venv_dir=Path("/proj/vllm-env"),
                            exists_fn=lambda _: False, which_fn=lambda _: None) is None


def test_find_vllm_binary_uses_windows_paths():
    found = find_vllm_binary(
        platform_name="Windows", venv_dir=Path("C:/proj/vllm-env"),
        exists_fn=lambda _: True, which_fn=lambda _: None,
    )
    assert found.endswith("Scripts/vllm.exe") or found.endswith("Scripts\\vllm.exe")


def test_find_vllm_binary_does_not_look_for_a_metal_venv_off_macos():
    probed = []
    find_vllm_binary(platform_name="Linux", venv_dir=Path("/proj/vllm-env"),
                     exists_fn=lambda path: probed.append(str(path)) or False,
                     which_fn=lambda _: None)
    assert not any(".venv-vllm-metal" in path for path in probed)
