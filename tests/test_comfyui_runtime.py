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
