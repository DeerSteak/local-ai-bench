import os
import sys

from scripts.runtime.shared import Shared


class _Process:
    def __init__(self, command, children=()):
        self.info = {"cmdline": command}
        self._children = list(children)
        self.terminated = False

    def children(self, recursive=False):
        assert recursive is True
        return self._children

    def terminate(self):
        self.terminated = True


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_qualification_stops_only_comfyui_from_its_managed_tree(tmp_path, monkeypatch):
    managed = tmp_path / "qualification-comfyui-runtime" / "ComfyUI"
    child = _Process(["worker"])
    matching = _Process(["python", str(managed / "main.py")], [child])
    unrelated = _Process(["python", "/opt/ComfyUI/main.py"])
    monkeypatch.setenv("LOCAL_AI_BENCH_QUALIFICATION", "1")

    assert Shared.stop_stale_qualification_comfyui(
        managed, [matching, unrelated],
    ) is True
    assert matching.terminated is True
    assert child.terminated is True
    assert unrelated.terminated is False


def test_normal_benchmark_never_stops_an_existing_comfyui(tmp_path, monkeypatch):
    process = _Process(["python", str(tmp_path / "ComfyUI" / "main.py")])
    monkeypatch.delenv("LOCAL_AI_BENCH_QUALIFICATION", raising=False)
    assert Shared.stop_stale_qualification_comfyui(tmp_path / "ComfyUI", [process]) is False
    assert process.terminated is False


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
