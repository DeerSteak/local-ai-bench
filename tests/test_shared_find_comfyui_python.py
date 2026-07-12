import os
import sys

from shared import Shared


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_prefers_windows_portable_python_embeded(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    comfyui_dir = tmp_path / "ComfyUI"
    portable = tmp_path / "python_embeded" / "python.exe"
    _touch(portable)
    # Also create a lower-priority candidate to confirm the portable build wins.
    _touch(comfyui_dir / "venv" / "bin" / "python")

    assert Shared.find_comfyui_python(comfyui_dir) == str(portable)


def test_falls_back_to_venv_bin_python_when_no_portable_build(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    comfyui_dir = tmp_path / "ComfyUI"
    venv_python = comfyui_dir / "venv" / "bin" / "python"
    _touch(venv_python)
    _touch(comfyui_dir / ".venv" / "bin" / "python")  # lower priority

    assert Shared.find_comfyui_python(comfyui_dir) == str(venv_python)


def test_falls_back_to_dotvenv_when_no_venv(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    comfyui_dir = tmp_path / "ComfyUI"
    dotvenv_python = comfyui_dir / ".venv" / "bin" / "python"
    _touch(dotvenv_python)

    assert Shared.find_comfyui_python(comfyui_dir) == str(dotvenv_python)


def test_falls_back_to_current_virtual_env_when_no_candidate_exists(tmp_path, monkeypatch):
    comfyui_dir = tmp_path / "ComfyUI"
    comfyui_dir.mkdir()
    venv_dir = tmp_path / "outer-venv"
    venv_python = venv_dir / "bin" / "python"
    _touch(venv_python)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))

    assert Shared.find_comfyui_python(comfyui_dir) == str(venv_python)


def test_falls_back_to_sys_executable_when_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    comfyui_dir = tmp_path / "ComfyUI"
    comfyui_dir.mkdir()

    assert Shared.find_comfyui_python(comfyui_dir) == sys.executable
