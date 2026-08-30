from pathlib import Path
import importlib
import sys


SETUP_CHECK = Path(__file__).parents[1] / "scripts" / "setup" / "setup_check.py"


def test_setup_coordinator_retains_install_and_summary_stages():
    source = SETUP_CHECK.read_text(encoding="utf-8")

    assert source.index("_host_error := qualification_host_error(") < source.index(
        "nvidia = discover_nvidia()"
    )
    assert "ensure_comfyui(" in source
    assert "provision_comfyui_assets(" in source
    assert "write_setup_config(" in source
    assert 'section("Summary")' in source


def test_gpu_prerequisite_failures_stop_before_llamacpp_install():
    source = SETUP_CHECK.read_text(encoding="utf-8")
    install_block = source[source.index("if intel_linux and not intel_linux_runtime:"):
                           source.index("req_file = SCRIPT_DIR / \"requirements.txt\"")]

    assert install_block.count("sys.exit(1)") == 5
    assert "issues.append" not in install_block


def test_post_install_xpu_probe_uses_oneapi_environment():
    source = SETUP_CHECK.read_text(encoding="utf-8")

    assert '_llamacpp_probe_env = oneapi_environment() if _required_llamacpp_backend == "xpu"' in source
    assert source.count("env=_llamacpp_probe_env") == 1
    assert '_post_install_probe_env = (' in source
    assert 'env=_post_install_probe_env,' in source
    assert 'context="setup"' in source


def test_native_nvidia_qualification_installs_driver_before_runtime_discovery():
    source = SETUP_CHECK.read_text(encoding="utf-8")
    driver_block = source[source.index("_initial_nvidia = discover_nvidia()"):
                          source.index("_initial_rocm = discover_rocm()")]

    assert "qualification_needs_native_nvidia_driver(" in driver_block
    assert "discover_linux_nvidia_gpu()" in driver_block
    assert "nouveau_loaded()" in driver_block
    assert "run_native_nvidia_driver_install(_driver_plan)" in driver_block
    assert "reboot to load it" in driver_block
    assert "sys.exit(NATIVE_NVIDIA_REBOOT_EXIT_CODE)" in driver_block


def test_qualification_repairs_cpu_only_gpu_runtimes():
    source = SETUP_CHECK.read_text(encoding="utf-8")

    assert "_llamacpp_backend_mismatch" in source
    assert "llamacpp_backend_rebuild_warning(" in source
    assert "vllm_runtime_expectations(\n        vllm_support.method," in source
    assert "recreate=_qualification_vllm_runtime_error is not None" in source


def test_native_nvidia_setup_requires_toolkit_before_source_build():
    source = SETUP_CHECK.read_text(encoding="utf-8")

    assert "native_cuda_toolkit_plan(" in source
    assert 'fail("CUDA toolkit installation did not provide nvcc")' in source
    assert "sys.exit(1)" in source[source.index("if _cuda_plan:"):
                                   source.index("vllm_note =")]


def test_setup_coordinator_import_is_safe():
    sys.modules.pop("scripts.setup.setup_check", None)
    module = importlib.import_module("scripts.setup.setup_check")

    assert callable(module.main)
