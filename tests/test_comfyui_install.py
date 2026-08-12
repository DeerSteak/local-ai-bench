from types import SimpleNamespace

import pytest

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


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _portable_release(*assets):
    return {"tag_name": "v1.0", "assets": list(assets)}


def _asset(name="ComfyUI_windows_portable_nvidia_cu128.7z"):
    return {"name": name, "size": 1024, "browser_download_url": f"https://example/{name}"}


def _install_portable(monkeypatch, tmp_path, release):
    monkeypatch.setattr(comfyui_install.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(comfyui_install.json, "load", lambda _response: release)
    logs = {name: [] for name in ("info", "warn", "fail", "ok")}
    result = comfyui_install.install_portable(
        tmp_path, "nvidia_cu", "NVIDIA", logs["info"].append,
        logs["warn"].append, logs["fail"].append, logs["ok"].append,
    )
    return result, logs


def test_portable_install_rejects_release_without_matching_asset(monkeypatch, tmp_path):
    result, logs = _install_portable(
        monkeypatch, tmp_path, _portable_release(_asset("ComfyUI_windows_portable_amd.7z")),
    )
    assert result is False
    assert logs["fail"] == ["No NVIDIA portable build found in latest ComfyUI release"]


@pytest.mark.parametrize("failure", [OSError("download failed"), ValueError("invalid archive")])
def test_portable_install_removes_archive_after_download_or_validation_failure(
        monkeypatch, tmp_path, failure):
    archive = tmp_path / _asset()["name"]

    def download(_url, destination, **_kwargs):
        destination.touch()
        if isinstance(failure, OSError):
            raise failure

    monkeypatch.setattr(comfyui_install, "download_file", download)
    monkeypatch.setattr(comfyui_install, "validate_7z_archive", lambda _path: (_ for _ in ()).throw(failure))
    result, logs = _install_portable(monkeypatch, tmp_path, _portable_release(_asset()))
    assert result is False
    assert not archive.exists()
    assert "Download or archive validation failed" in logs["fail"][0]


def test_portable_install_downloads_7zr_when_no_extractor_is_on_path(monkeypatch, tmp_path):
    archive = tmp_path / _asset()["name"]
    calls = []

    def download(url, destination, **_kwargs):
        calls.append((url, destination))
        destination.touch()

    def run(command, **_kwargs):
        wrapper = tmp_path / "ComfyUI_windows_portable"
        wrapper.mkdir()
        (wrapper / "ComfyUI").mkdir()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(comfyui_install, "download_file", download)
    monkeypatch.setattr(comfyui_install, "validate_7z_archive", lambda _path: None)
    monkeypatch.setattr(comfyui_install.shutil, "which", lambda _name: None)
    monkeypatch.setattr(comfyui_install.subprocess, "run", run)
    result, _logs = _install_portable(monkeypatch, tmp_path, _portable_release(_asset()))
    assert result is True
    assert calls[0][1] == archive
    assert calls[1][1] == tmp_path / "7zr.exe"
    assert (tmp_path / "ComfyUI").is_dir()
    assert not archive.exists()


def test_portable_install_stops_when_7zr_download_fails(monkeypatch, tmp_path):
    archive = tmp_path / _asset()["name"]

    def download(_url, destination, **_kwargs):
        destination.touch()
        if destination.name == "7zr.exe":
            raise OSError("offline")

    monkeypatch.setattr(comfyui_install, "download_file", download)
    monkeypatch.setattr(comfyui_install, "validate_7z_archive", lambda _path: None)
    monkeypatch.setattr(comfyui_install.shutil, "which", lambda _name: None)
    result, logs = _install_portable(monkeypatch, tmp_path, _portable_release(_asset()))
    assert result is False
    assert not archive.exists()
    assert not (tmp_path / "7zr.exe").exists()
    assert logs["fail"] == ["Could not download 7zr.exe: offline"]


def test_portable_install_removes_archive_after_extraction_failure(monkeypatch, tmp_path):
    archive = tmp_path / _asset()["name"]
    monkeypatch.setattr(
        comfyui_install, "download_file",
        lambda _url, destination, **_kwargs: destination.touch(),
    )
    monkeypatch.setattr(comfyui_install, "validate_7z_archive", lambda _path: None)
    monkeypatch.setattr(comfyui_install.shutil, "which", lambda _name: "/usr/bin/7z")
    monkeypatch.setattr(
        comfyui_install.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stderr="corrupt"),
    )
    result, logs = _install_portable(monkeypatch, tmp_path, _portable_release(_asset()))
    assert result is False
    assert not archive.exists()
    assert logs["fail"] == ["Extraction failed:\ncorrupt"]


def test_flatten_portable_replaces_only_colliding_top_level_entries(tmp_path):
    wrapper = tmp_path / "ComfyUI_windows_portable"
    wrapper.mkdir()
    incoming_dir = wrapper / "ComfyUI"
    incoming_dir.mkdir()
    (incoming_dir / "new.txt").write_text("new")
    (wrapper / "run.bat").write_text("new runner")
    existing_dir = tmp_path / "ComfyUI"
    existing_dir.mkdir()
    (existing_dir / "old.txt").write_text("old")
    (tmp_path / "run.bat").write_text("old runner")
    untouched = tmp_path / "keep.txt"
    untouched.write_text("keep")

    comfyui_install._flatten_portable(tmp_path)

    assert (tmp_path / "ComfyUI" / "new.txt").read_text() == "new"
    assert not (tmp_path / "ComfyUI" / "old.txt").exists()
    assert (tmp_path / "run.bat").read_text() == "new runner"
    assert untouched.read_text() == "keep"
    assert not wrapper.exists()


def test_ensure_cuda_arch_warns_when_probe_fails(monkeypatch, tmp_path):
    warnings = []
    monkeypatch.setattr(
        comfyui_install.subprocess, "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("probe failed")),
    )
    comfyui_install.ensure_cuda_arch(
        tmp_path / "python.exe", "8.9", [], _log, warnings.append, _log, _log,
    )
    assert warnings == ["Could not check torch CUDA architecture support: probe failed"]


def test_ensure_cuda_arch_reinstalls_for_missing_architecture(monkeypatch, tmp_path):
    commands, issues, failures = [], [], []
    monkeypatch.setattr(comfyui_install.subprocess, "check_output", lambda *_args, **_kwargs: "sm_80")
    monkeypatch.setattr(
        comfyui_install.subprocess, "run",
        lambda command: commands.append(command) or SimpleNamespace(returncode=1),
    )
    python = tmp_path / "python.exe"
    comfyui_install.ensure_cuda_arch(
        python, "8.9", issues, _log, _log, failures.append, _log,
    )
    assert commands[0][:5] == [str(python), "-s", "-m", "pip", "install"]
    assert failures == ["torch reinstall failed"]
    assert issues == [" ".join(commands[0])]
