from types import SimpleNamespace

from scripts.setup import comfyui_install


def _log(_message):
    pass


def test_existing_source_install_is_accepted(tmp_path):
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    assert comfyui_install.ensure(
        comfyui, tmp_path, "Linux", None, compute_capability=None,
        issues=[], info=_log, warn=_log, fail=_log, ok=_log,
    )


def test_missing_source_install_is_cloned(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        comfyui_install.subprocess, "run",
        lambda command, **_kwargs: calls.append(command) or SimpleNamespace(returncode=0),
    )
    assert comfyui_install.ensure(
        tmp_path / "ComfyUI", tmp_path, "Linux", None, compute_capability=None,
        issues=[], info=_log, warn=_log, fail=_log, ok=_log,
    )
    assert calls[0][:2] == ["git", "clone"]


def test_existing_windows_install_requires_portable_python(tmp_path):
    comfyui = tmp_path / "ComfyUI"
    comfyui.mkdir()
    issues = []
    assert not comfyui_install.ensure(
        comfyui, tmp_path, "Windows", "amd", compute_capability=None,
        issues=issues, info=_log, warn=_log, fail=_log, ok=_log,
    )
    assert issues
