from pathlib import Path

from scripts.setup.setup_check import accessible_file


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
