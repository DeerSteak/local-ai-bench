from pathlib import Path

import pytest

from scripts.release.qualification_runtime import (
    export_verified_bundle, managed_path, managed_processes, require_runtime_version, smoke_runtime,
    restore_runtime, runtime_path, snapshot_runtime, uninstall,
)


def repo(tmp_path):
    (tmp_path / "README.md").touch()
    (tmp_path / "scripts").mkdir()
    return tmp_path


def test_runtime_snapshot_and_restore_are_confined_to_clone(tmp_path):
    root = repo(tmp_path)
    runtime = root / "llama.cpp"
    runtime.mkdir()
    (runtime / "version.txt").write_text("baseline")
    snapshot_runtime(root, "llamacpp")
    (runtime / "version.txt").write_text("upgrade")
    restore_runtime(root, "llamacpp")
    assert (runtime / "version.txt").read_text() == "baseline"


def test_managed_paths_reject_unknown_names_and_symlinks(tmp_path):
    root = repo(tmp_path)
    with pytest.raises(ValueError, match="does not manage"):
        managed_path(root, "results")
    (root / "models").symlink_to(root / "outside")
    with pytest.raises(ValueError, match="symbolic links"):
        managed_path(root, "models")


def test_uninstall_removes_only_qualification_assets(tmp_path):
    root = repo(tmp_path)
    for name in ("llama.cpp", "qualification-runtime-baseline", "models"):
        (root / name).mkdir()
    evidence = root / "qualification-evidence"
    evidence.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "scripts.release.qualification_runtime.managed_processes", lambda _root: [],
        )
        uninstall(root, "llamacpp")
    assert not runtime_path(root, "llamacpp").exists()
    assert evidence.is_dir()


def test_bundle_export_is_immediately_verified(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "scripts.release.qualification_runtime.export_result_bundle",
        lambda *args, **kwargs: calls.append(("export", args, kwargs)),
    )
    monkeypatch.setattr(
        "scripts.release.qualification_runtime.verify_result_bundle",
        lambda path: calls.append(("verify", path)),
    )
    export_verified_bundle(tmp_path / "result.json", tmp_path / "bundle.zip", "machine")
    assert [call[0] for call in calls] == ["export", "verify"]


def test_discovery_rejects_a_different_runtime_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.release.qualification_runtime.runtime_version", lambda *_args: "0.26.0",
    )
    with pytest.raises(ValueError, match="expected vllm 0.27.1"):
        require_runtime_version(tmp_path, "vllm", "0.27.1")


def test_process_inspection_finds_only_clone_owned_commands(tmp_path):
    from types import SimpleNamespace
    root = repo(tmp_path)
    output = f"101 /usr/bin/python {root}/llama.cpp/server.py\n102 /usr/bin/python /other/app.py\n"
    found = managed_processes(
        root, run=lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output),
    )
    assert found == [f"101 /usr/bin/python {root}/llama.cpp/server.py"]


def test_upgraded_runtime_smoke_uses_quick_selected_engine(monkeypatch, tmp_path):
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(
        "scripts.release.qualification_runtime.subprocess.run",
        lambda command: calls.append(command) or SimpleNamespace(returncode=0),
    )
    smoke_runtime("vllm", tmp_path / "upgraded.json")
    assert "--quick" in calls[0]
    assert calls[0][calls[0].index("--engine") + 1] == "vllm"
    assert "--ack-experimental-engine" in calls[0]
