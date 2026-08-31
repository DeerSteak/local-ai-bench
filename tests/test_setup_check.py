from pathlib import Path

from scripts.setup.setup_check import (
    accessible_file, llamacpp_backend_rebuild_warning, llamacpp_install_action,
    running_comfyui_checkpoints_visible, vulkan_install_backend_flags,
)


def test_vulkan_install_never_combines_sycl_and_vulkan_selection():
    assert vulkan_install_backend_flags(False, False) == {
        "nvidia": False, "rocm": False, "intel_xpu": False, "vulkan": True,
    }


def test_accessible_file_accepts_a_regular_file(tmp_path):
    path = tmp_path / "python"
    path.touch()
    assert accessible_file(path)


def test_accessible_file_rejects_missing_and_permission_denied_paths(tmp_path, monkeypatch):
    assert not accessible_file(tmp_path / "missing")

    def denied(_path):
        raise PermissionError("mounted path is not traversable")

    monkeypatch.setattr(Path, "is_file", denied)
    assert not accessible_file(tmp_path / "inaccessible")


def test_backend_rebuild_warning_distinguishes_setup_from_qualification():
    assert llamacpp_backend_rebuild_warning(
        "vulkan", "cuda", qualification=False,
    ) == "llama-server exposes vulkan, but setup requires cuda — it will be rebuilt"
    assert llamacpp_backend_rebuild_warning(
        None, "xpu", qualification=True,
    ) == ("llama-server exposes no detectable backend, but qualification requires xpu "
          "— it will be rebuilt")


def test_intel_llamacpp_failure_retries_the_managed_sycl_build(tmp_path):
    runtime = tmp_path / "llama.cpp"

    action = llamacpp_install_action("xpu", "cmake configure failed", runtime)

    assert "managed llama.cpp Intel oneAPI/SYCL build" in action
    assert str(runtime) in action
    assert "Last failure: cmake configure failed" in action
    assert "No manual llama.cpp installation is required" in action


def test_running_comfyui_visibility_treats_an_unknown_loader_as_unavailable():
    model = {"checkpoint": "future.safetensors", "checkpoint_loader": "FutureLoader"}
    response = type("Response", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *_args: None,
        "read": lambda self: b"{}",
    })()
    assert running_comfyui_checkpoints_visible(
        [model], {"future.safetensors"}, urlopen=lambda *_args, **_kwargs: response,
    ) is None
