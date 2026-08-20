from types import SimpleNamespace

from scripts.setup import comfyui_runtime


def _log(_message):
    pass


def test_prepare_skips_missing_installation(tmp_path):
    assert not comfyui_runtime.prepare(
        tmp_path / "missing", tmp_path / "models", tmp_path / "paths.yaml",
        portable_python=tmp_path / "python.exe", intel_xpu=False, rocm=False,
        issues=[], info=_log, warn=_log, fail=_log, ok=_log,
    )


def test_prepare_installs_missing_requirements(monkeypatch, tmp_path):
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    (comfyui / "requirements.txt").write_text("aiohttp\n", encoding="utf-8")
    calls = []
    return_codes = iter([1, 0])
    monkeypatch.setattr(comfyui_runtime, "find_comfyui_python", lambda _path: "python")
    monkeypatch.setattr(
        comfyui_runtime.subprocess, "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(
            returncode=next(return_codes), stdout="",
        ),
    )
    monkeypatch.setattr(comfyui_runtime, "write_extra_model_paths", lambda *_args: None)
    monkeypatch.setattr(comfyui_runtime, "legacy_models_dir_with_assets", lambda _path: None)
    monkeypatch.setattr(comfyui_runtime, "add_managed_models_to_comfyui", lambda *_args: "paths.yaml")

    assert comfyui_runtime.prepare(
        comfyui, tmp_path / "models", tmp_path / "paths.yaml",
        portable_python=tmp_path / "python.exe", intel_xpu=False, rocm=False,
        issues=[], info=_log, warn=_log, fail=_log, ok=_log,
    )
    assert calls[1] == ["python", "-m", "pip", "install", "-r", str(comfyui / "requirements.txt")]


def test_torch_backend_probe_uses_selected_comfyui_python():
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="6.4.43482\n")

    assert comfyui_runtime.torch_backend_available("comfy-python", "ROCm", run=run)
    assert calls == [[
        "comfy-python", "-c",
        "import torch; assert torch.version.hip; torch.zeros(1, device='cuda'); "
        "print(torch.version.hip)",
    ]]


def test_wrong_torch_backend_is_force_reinstalled_and_verified(monkeypatch):
    probes = iter([False, True])
    installs = []
    monkeypatch.setattr(
        comfyui_runtime, "torch_backend_available",
        lambda python, marker: next(probes),
    )
    monkeypatch.setattr(
        comfyui_runtime.subprocess, "run",
        lambda command: installs.append(command) or SimpleNamespace(returncode=0),
    )
    issues = []

    comfyui_runtime._ensure_torch_backend(
        "comfy-python", "rocm6.4", "ROCm", issues, _log, _log, _log,
    )

    assert issues == []
    assert installs == [[
        "comfy-python", "-m", "pip", "install", "--upgrade", "--force-reinstall",
        "--index-url", "https://download.pytorch.org/whl/rocm6.4",
        "torch", "torchvision", "torchaudio",
    ]]


def test_backend_install_is_rejected_when_runtime_probe_still_fails(monkeypatch):
    monkeypatch.setattr(comfyui_runtime, "torch_backend_available", lambda *_args: False)
    monkeypatch.setattr(
        comfyui_runtime.subprocess, "run", lambda _command: SimpleNamespace(returncode=0),
    )
    issues = []

    comfyui_runtime._ensure_torch_backend(
        "comfy-python", "rocm6.4", "ROCm", issues, _log, _log, _log,
    )

    assert issues and "--force-reinstall" in issues[0]


def test_rocm_72_wsl_uses_amds_pinned_python312_wheels():
    command = comfyui_runtime.rocm_torch_install_command(
        "comfy-python", (7, 2), wsl=True,
    )
    assert "numpy==1.26.4" in command
    assert "scipy==1.16.3" in command
    assert all("rocm-rel-7.2/" in item for item in command if item.startswith("https://"))
    assert any("torch-2.9.1%2Brocm7.2.0" in item for item in command)
    assert any("triton-3.5.1%2Brocm7.2.0" in item for item in command)
    assert "download.pytorch.org" not in " ".join(command)


def test_rocm_72_native_uses_amds_pinned_721_wheels():
    command = comfyui_runtime.rocm_torch_install_command(
        "comfy-python", (7, 2), wsl=False,
    )
    assert all("rocm-rel-7.2.1/" in item for item in command if item.startswith("https://"))
    assert any("torch-2.9.1%2Brocm7.2.1" in item for item in command)


def test_older_rocm_keeps_the_generic_pytorch_index():
    command = comfyui_runtime.rocm_torch_install_command(
        "comfy-python", (6, 4), wsl=False,
    )
    assert "https://download.pytorch.org/whl/rocm6.4" in command


def test_wsl_hsa_runtime_removal_uses_selected_comfyui_python():
    calls = []
    assert comfyui_runtime.remove_wsl_bundled_hsa_runtime(
        "comfy-python",
        run=lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert calls[0][:2] == ["comfy-python", "-c"]
    assert "libhsa-runtime64.so*" in calls[0][2]


def test_rocm_dependency_probe_uses_selected_comfyui_python():
    calls = []
    assert comfyui_runtime.rocm_python_dependencies_available(
        "comfy-python",
        run=lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert calls[0][:2] == ["comfy-python", "-c"]
    assert "numpy" in calls[0][2]
    assert "scipy" in calls[0][2]


def test_rocm_wheel_install_removes_wsl_hsa_runtime_then_verifies(monkeypatch):
    probes = iter([False, True])
    events = []
    monkeypatch.setattr(
        comfyui_runtime, "torch_backend_available",
        lambda *_args: events.append("probe") or next(probes),
    )
    monkeypatch.setattr(
        comfyui_runtime.subprocess, "run",
        lambda _command: events.append("install") or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        comfyui_runtime, "remove_wsl_bundled_hsa_runtime",
        lambda _python: events.append("hsa") or True,
    )
    monkeypatch.setattr(
        comfyui_runtime, "rocm_python_dependencies_available",
        lambda _python: events.append("dependencies") or True,
    )
    issues = []

    comfyui_runtime._ensure_rocm_torch_backend(
        "comfy-python", (7, 2), wsl=True, issues=issues,
        info=_log, fail=_log, ok=_log,
    )

    assert issues == []
    assert events == ["probe", "install", "hsa", "dependencies", "probe"]


def test_existing_rocm_torch_repairs_only_incompatible_python_dependencies(monkeypatch):
    dependency_probes = iter([False, True])
    installs = []
    monkeypatch.setattr(comfyui_runtime, "torch_backend_available", lambda *_args: True)
    monkeypatch.setattr(
        comfyui_runtime, "rocm_python_dependencies_available",
        lambda _python: next(dependency_probes),
    )
    monkeypatch.setattr(
        comfyui_runtime.subprocess, "run",
        lambda command: installs.append(command) or SimpleNamespace(returncode=0),
    )
    issues = []

    comfyui_runtime._ensure_rocm_torch_backend(
        "comfy-python", (7, 2), wsl=True, issues=issues,
        info=_log, fail=_log, ok=_log,
    )

    assert issues == []
    assert installs == [[
        "comfy-python", "-m", "pip", "install", "--upgrade", "--force-reinstall",
        "numpy==1.26.4", "scipy==1.16.3",
    ]]


def test_prepare_installs_rocm_torch_before_comfyui_requirements(monkeypatch, tmp_path):
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    (comfyui / "requirements.txt").write_text("torch\naiohttp\n", encoding="utf-8")
    events = []
    monkeypatch.setattr(comfyui_runtime, "find_comfyui_python", lambda _path: "python")
    monkeypatch.setattr(
        comfyui_runtime, "_ensure_rocm_torch_backend",
        lambda *_args, **_kwargs: events.append("rocm"),
    )
    monkeypatch.setattr(
        comfyui_runtime.subprocess, "run",
        lambda command, **_kwargs: events.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(comfyui_runtime, "write_extra_model_paths", lambda *_args: None)
    monkeypatch.setattr(comfyui_runtime, "legacy_models_dir_with_assets", lambda _path: None)
    monkeypatch.setattr(
        comfyui_runtime, "add_managed_models_to_comfyui", lambda *_args: "paths.yaml",
    )

    comfyui_runtime.prepare(
        comfyui, tmp_path / "models", tmp_path / "paths.yaml",
        portable_python=tmp_path / "python.exe", intel_xpu=False, rocm=True,
        rocm_version=(7, 2), wsl=True,
        issues=[], info=_log, warn=_log, fail=_log, ok=_log,
    )

    assert events[0] == "rocm"
    assert events[1][:5] == ["python", "-m", "pip", "show", "aiohttp"]
