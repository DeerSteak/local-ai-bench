from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.setup.directory_transaction import DirectorySwapError, swap_staged_directory


@dataclass(frozen=True)
class Validation:
    success: bool
    detail: str = ""
    version: str | None = None


def test_swap_staged_directory_commits_and_removes_backup(tmp_path):
    target = tmp_path / "runtime"
    staged = tmp_path / "staged"
    backup = tmp_path / "backup"
    target.mkdir()
    staged.mkdir()
    (target / "old").touch()
    (staged / "new").touch()

    outcome = swap_staged_directory(
        target, staged, backup, had_target=True,
        validate=lambda path: Validation(success=(path / "new").is_file()),
        replace=lambda source, destination: Path(source).replace(destination),
        remove=lambda path: __import__("shutil").rmtree(path),
    )

    assert outcome.validation is not None
    assert outcome.validation.success
    assert (target / "new").is_file()
    assert not backup.exists()


def test_swap_staged_directory_rolls_back_failed_validation(tmp_path):
    target = tmp_path / "runtime"
    staged = tmp_path / "staged"
    backup = tmp_path / "backup"
    target.mkdir()
    staged.mkdir()
    (target / "old").touch()

    with pytest.raises(DirectorySwapError, match="invalid runtime"):
        swap_staged_directory(
            target, staged, backup, had_target=True,
            validate=lambda _path: Validation(success=False, detail="invalid runtime"),
            replace=lambda source, destination: Path(source).replace(destination),
            remove=lambda path: __import__("shutil").rmtree(path),
        )

    assert (target / "old").is_file()
    assert not backup.exists()


def test_swap_staged_directory_supports_first_install(tmp_path):
    target = tmp_path / "runtime"
    staged = tmp_path / "staged"
    backup = tmp_path / "backup"
    staged.mkdir()

    swap_staged_directory(
        target, staged, backup, had_target=False,
        replace=lambda source, destination: Path(source).replace(destination),
        remove=lambda path: __import__("shutil").rmtree(path),
    )

    assert target.is_dir()
    assert not backup.exists()
