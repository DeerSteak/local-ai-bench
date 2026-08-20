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
