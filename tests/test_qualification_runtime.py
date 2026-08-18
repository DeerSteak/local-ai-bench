from pathlib import Path

import pytest

from scripts.release.qualification_runtime import (
    managed_path, restore_runtime, runtime_path, snapshot_runtime, uninstall,
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
    uninstall(root, "llamacpp")
    assert not runtime_path(root, "llamacpp").exists()
    assert evidence.is_dir()
