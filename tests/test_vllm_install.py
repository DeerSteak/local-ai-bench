import sys
from pathlib import Path

from scripts.setup.vllm_install import (
    ROCM_WHEEL_INDEX,
    VLLM_ROCM_WHEEL_TARGETS,
    NIGHTLY_CU130_INDEX,
    hf_cache_model_complete,
    hf_cache_model_dir,
    vllm_cache_home,
    build_tools_command,
    missing_build_tools,
    missing_python_headers,
    python_dev_package_command,
    python_version_from_include_dir,
    find_vllm_binary,
    find_vllm_launcher,
    parse_launcher_extra_args,
    read_launcher_extra_args,
    find_vllm_server,
    vllm_server_reachable,
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


def test_strix_halo_gfx1151_is_experimental_not_supported():
    result = support(rocm_ok=True, rocm_version=(7, 0), rocm_gfx_targets=["gfx1151"])
    assert result.status == "experimental"
    assert result.method == "rocm_wheel"  # still installable, but the user is warned
    assert "gfx1151" in result.reason
    assert "amd-strix-halo-vllm-toolboxes" in result.reason


def test_every_wheel_target_stays_supported():
    for target in VLLM_ROCM_WHEEL_TARGETS:
        result = support(rocm_ok=True, rocm_version=(7, 0), rocm_gfx_targets=[target])
        assert result.status == "supported", target


def test_an_unknown_gfx_target_does_not_downgrade_a_targeted_gpu():
    result = support(rocm_ok=True, rocm_version=(7, 0),
                     rocm_gfx_targets=["gfx942", "gfx1151"])
    assert result.status == "supported", "one targeted GPU is enough to use the wheels"


def test_undetected_gfx_targets_keep_the_default_supported_verdict():
    assert support(rocm_ok=True, rocm_version=(7, 0), rocm_gfx_targets=None).status == "supported"
    assert support(rocm_ok=True, rocm_version=(7, 0), rocm_gfx_targets=[]).status == "supported"


def test_gfx_gate_does_not_override_an_unsupported_rocm_version():
    result = support(rocm_ok=True, rocm_version=(6, 0), rocm_gfx_targets=["gfx1151"])
    assert result.status == "unsupported"
    assert "6.0" in result.reason


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


def test_apple_silicon_is_unsupported():
    result = support(os_name="Darwin", machine="arm64")
    assert (result.status, result.method) == ("unsupported", None)


def test_intel_mac_is_unsupported():
    result = support(os_name="Darwin", machine="x86_64")
    assert (result.status, result.method) == ("unsupported", None)


def test_windows_is_unsupported_even_with_an_nvidia_gpu():
    result = support(os_name="Windows", nvidia_ok=True, compute_cap="8.9")
    assert result.status == "unsupported"
    assert "WSL2" in result.reason
    assert "setup.md" in result.reason


def test_wsl2_takes_the_linux_path_since_it_reports_as_linux():
    result = support(os_name="Linux", nvidia_ok=True, compute_cap="8.9")
    assert result.status == "supported"
    assert result.method == "cuda_wheel"


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
    assert pip == ["/v/bin/python", "-m", "pip", "install", "vllm[bench]"]
    assert "--torch-backend=auto" not in pip  # a pip-only flag would fail the install


def test_every_install_method_requests_the_bench_extra():
    """`vllm bench` deps ship only with the extra, and the vllmbench test needs them."""
    for method in ("cuda_wheel", "rocm_wheel", "nightly_cu130"):
        for uv_available in (True, False):
            command = vllm_install_command(method, "/v/bin/python", uv_available=uv_available)
            assert "vllm[bench]" in command
            assert "vllm" not in command


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


def test_find_vllm_binary_returns_none_when_nothing_is_installed():
    assert find_vllm_binary(platform_name="Linux", venv_dir=Path("/proj/vllm-env"),
                            exists_fn=lambda _: False, which_fn=lambda _: None) is None


def test_find_vllm_binary_uses_windows_paths():
    found = find_vllm_binary(
        platform_name="Windows", venv_dir=Path("C:/proj/vllm-env"),
        exists_fn=lambda _: True, which_fn=lambda _: None,
    )
    assert found is not None
    assert found.endswith("Scripts/vllm.exe") or found.endswith("Scripts\\vllm.exe")


# ── already-running server ──

class _FakeResponse:
    def __init__(self, status): self.status = status
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def test_server_probe_accepts_a_healthy_endpoint():
    seen = {}
    def opener(url, timeout=None):
        seen["url"] = url
        return _FakeResponse(200)
    assert vllm_server_reachable("http://localhost:8000", open_fn=opener) is True
    assert seen["url"] == "http://localhost:8000/v1/models"


def test_server_probe_rejects_an_error_status():
    assert vllm_server_reachable("http://localhost:8000",
                                 open_fn=lambda *a, **k: _FakeResponse(503)) is False


def test_server_probe_is_false_when_nothing_is_listening():
    def refuse(*args, **kwargs):
        raise OSError("connection refused")
    assert vllm_server_reachable("http://localhost:8000", open_fn=refuse) is False


def test_server_discovery_checks_amd_launch_port_as_well_as_the_default():
    tried = []
    def opener(url, timeout=None):
        tried.append(url)
        if ":8001/" in url:
            return _FakeResponse(200)
        raise OSError("connection refused")
    assert find_vllm_server(open_fn=opener) == "http://localhost:8001"
    assert tried == ["http://localhost:8000/v1/models", "http://localhost:8001/v1/models"]


def test_server_discovery_prefers_the_first_port_that_answers():
    assert find_vllm_server(open_fn=lambda *a, **k: _FakeResponse(200)) == "http://localhost:8000"


def test_server_discovery_returns_none_when_no_port_answers():
    def refuse(*args, **kwargs):
        raise OSError("connection refused")
    assert find_vllm_server(open_fn=refuse) is None


def test_server_discovery_accepts_an_explicit_port_list():
    seen = []
    def opener(url, timeout=None):
        seen.append(url)
        raise OSError("nope")
    assert find_vllm_server(ports=[9999], open_fn=opener) is None
    assert seen == ["http://localhost:9999/v1/models"]


# ── platform launcher ──

def test_find_vllm_launcher_prefers_a_platform_wrapper():
    assert find_vllm_launcher(which_fn=lambda name:
        "/usr/bin/vllm-launch" if name == "vllm-launch" else None) == "/usr/bin/vllm-launch"
    assert find_vllm_launcher(which_fn=lambda _: None) is None


def test_launcher_conf_extra_args_are_parsed():
    assert parse_launcher_extra_args(
        "VLLM_EXTRA_ARGS=(--gpu-memory-utilization 0.85 --enforce-eager)"
    ) == ["--gpu-memory-utilization", "0.85", "--enforce-eager"]


def test_launcher_conf_append_form_accumulates():
    text = "VLLM_EXTRA_ARGS=(--a 1)\nVLLM_EXTRA_ARGS+=(--b 2)\n"
    assert parse_launcher_extra_args(text) == ["--a", "1", "--b", "2"]


def test_launcher_conf_respects_quoting_and_ignores_other_lines():
    text = '# comment\nOTHER=(--x)\nVLLM_EXTRA_ARGS=(--served-model-name "two words")\n'
    assert parse_launcher_extra_args(text) == ["--served-model-name", "two words"]


def test_launcher_conf_parsing_is_empty_for_junk():
    for text in ("", None, "VLLM_EXTRA_ARGS=", "nothing"):
        assert parse_launcher_extra_args(text) == []


def test_reading_a_missing_launcher_conf_is_not_an_error(tmp_path):
    assert read_launcher_extra_args(tmp_path / "nope.conf") == []


def test_reading_a_real_launcher_conf(tmp_path):
    conf = tmp_path / "vllm-launch.conf"
    conf.write_text("VLLM_EXTRA_ARGS=(--max-model-len 8192)\n")
    assert read_launcher_extra_args(conf) == ["--max-model-len", "8192"]


# ── model cache location ──

def test_cache_home_prefers_the_launchers_own_cache():
    found = vllm_cache_home(launcher="/usr/bin/vllm-launch", env={},
                            exists_fn=lambda path: path.name == "models")
    assert found == Path("~/.local/share/vLLM/models").expanduser()


def test_cache_home_falls_back_when_the_launcher_cache_is_absent():
    assert vllm_cache_home(launcher="/usr/bin/vllm-launch", env={},
                           exists_fn=lambda _: False) == Path("~/.cache/huggingface").expanduser()


def test_cache_home_without_a_launcher_uses_the_standard_cache():
    assert vllm_cache_home(launcher=None, env={},
                           exists_fn=lambda _: True) == Path("~/.cache/huggingface").expanduser()


def test_cache_home_honours_hf_home():
    assert vllm_cache_home(launcher=None, env={"HF_HOME": "/mnt/hf"},
                           exists_fn=lambda _: False) == Path("/mnt/hf")


def test_hf_cache_model_dir_uses_the_hub_naming_convention():
    assert hf_cache_model_dir(Path("/c"), "cyankiwi/granite-4.1-3b-AWQ-INT4") == \
        Path("/c/hub/models--cyankiwi--granite-4.1-3b-AWQ-INT4")


def test_cache_completeness_requires_weights_and_config_in_one_snapshot(tmp_path):
    repo = "org/model"
    snapshot = hf_cache_model_dir(tmp_path, repo) / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    assert hf_cache_model_complete(tmp_path, repo) is False

    (snapshot / "model.safetensors").touch()
    assert hf_cache_model_complete(tmp_path, repo) is False, "weights without config are unloadable"

    (snapshot / "config.json").touch()
    assert hf_cache_model_complete(tmp_path, repo) is True


def test_cache_completeness_is_false_for_an_unfetched_repo(tmp_path):
    assert hf_cache_model_complete(tmp_path, "org/never-downloaded") is False


def test_cache_completeness_ignores_a_partial_sibling_snapshot(tmp_path):
    repo = "org/model"
    snapshots = hf_cache_model_dir(tmp_path, repo) / "snapshots"
    (snapshots / "partial").mkdir(parents=True)
    (snapshots / "partial" / "config.json").touch()
    assert hf_cache_model_complete(tmp_path, repo) is False

    good = snapshots / "complete"
    good.mkdir()
    (good / "config.json").touch()
    (good / "model.safetensors").touch()
    assert hf_cache_model_complete(tmp_path, repo) is True


# ── Python development headers (Triton JIT dependency) ──

def test_missing_headers_are_reported_with_the_expected_path():
    assert missing_python_headers("/usr/include/python3.12", exists_fn=lambda _: False) == \
        "/usr/include/python3.12/Python.h"
    assert missing_python_headers("/usr/include/python3.12", exists_fn=lambda _: True) is None


def test_an_unknown_include_dir_is_not_reported_as_missing():
    """An interpreter we could not query is not evidence of a broken system."""
    assert missing_python_headers(None) is None
    assert missing_python_headers("") is None


def test_dev_package_commands_per_manager():
    which = lambda name: f"/usr/bin/{name}"
    apt = python_dev_package_command("apt-get", (3, 12), which_fn=which)
    dnf = python_dev_package_command("dnf", (3, 12), which_fn=which)
    zypper = python_dev_package_command("zypper", (3, 12), which_fn=which)
    assert apt is not None and apt[-1] == "python3.12-dev"
    assert dnf is not None and dnf[-1] == "python3-devel"
    assert zypper is not None and zypper[-1] == "python312-devel"


def test_dev_package_command_is_none_without_that_manager():
    assert python_dev_package_command("apt-get", (3, 12), which_fn=lambda _: None) is None


def test_unknown_package_managers_are_declined():
    assert python_dev_package_command("brew", (3, 12), which_fn=lambda n: f"/usr/bin/{n}") is None
    assert python_dev_package_command("pacman", (3, 12), which_fn=lambda n: f"/usr/bin/{n}") is None


def test_dev_package_command_is_noninteractive():
    command = python_dev_package_command("apt-get", (3, 12), which_fn=lambda n: f"/usr/bin/{n}")
    assert command is not None
    assert "-y" in command, "setup must not stall on an apt confirmation prompt"


def test_header_package_targets_the_venv_interpreter_not_setups_own():
    """setup runs on bench-env's Python; the vLLM venv is often a different minor
    version, and installing headers for the wrong one fixes nothing."""
    version = python_version_from_include_dir("/usr/include/python3.12")
    assert version == (3, 12)
    assert version is not None
    command = python_dev_package_command("apt-get", version, which_fn=lambda n: f"/usr/bin/{n}")
    assert command is not None and command[-1] == "python3.12-dev"


def test_include_dir_version_parsing_handles_other_layouts():
    assert python_version_from_include_dir("/usr/include/python3.9") == (3, 9)
    assert python_version_from_include_dir("/opt/py/include/python3.13/") == (3, 13)
    assert python_version_from_include_dir("/opt/weird/include") is None
    assert python_version_from_include_dir(None) is None


# ── JIT build tools ──

def test_ninja_is_reported_missing_from_a_bare_venv():
    """FlashInfer shells out to ninja when compiling sampling kernels."""
    assert missing_build_tools(Path("/v"), exists_fn=lambda _: False) == ["ninja"]
    assert missing_build_tools(Path("/v"), exists_fn=lambda _: True) == []


def test_build_tools_install_into_the_venv_without_sudo():
    command = build_tools_command("/v/bin/python", ["ninja"])
    assert command == ["/v/bin/python", "-m", "pip", "install", "ninja"]
    assert command is not None and "sudo" not in command
    assert build_tools_command("/v/bin/python", []) is None


def test_build_tools_are_looked_for_in_the_venv_bin():
    probed = []
    missing_build_tools(Path("/v"), exists_fn=lambda path: probed.append(path) or True)
    assert all(path.parent.name in ("bin", "Scripts") for path in probed)
