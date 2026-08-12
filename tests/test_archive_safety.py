import stat
import tarfile
import io
import os
import zipfile

import pytest

from scripts.setup.archive_safety import safe_extract_tar, safe_extract_zip, validate_archive_names


def test_archive_name_validation_accepts_normal_relative_paths():
    assert validate_archive_names(["folder/file.txt", "bin/tool.exe"]) == (
        "folder/file.txt", "bin/tool.exe",
    )


@pytest.mark.parametrize("name", [
    "../outside", "folder/../../outside", "/absolute", "C:/absolute", "C:\\absolute",
    "folder\\..\\outside", "./ambiguous", "folder//ambiguous", "bad\x00name",
])
def test_archive_name_validation_rejects_escaping_or_ambiguous_paths(name):
    with pytest.raises(ValueError):
        validate_archive_names([name])


def test_archive_name_validation_rejects_normalized_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_archive_names(["folder/file", "folder\\file"])


def test_safe_zip_extracts_regular_files(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("folder/", "")
        output.writestr("folder/file.txt", "content")
    destination = tmp_path / "destination"
    safe_extract_zip(archive, destination)
    assert (destination / "folder" / "file.txt").read_text(encoding="utf-8") == "content"


def test_safe_zip_rejects_traversal_before_writing_any_member(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("written-first.txt", "must not appear")
        output.writestr("../outside.txt", "escape")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="escapes"):
        safe_extract_zip(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_safe_zip_rejects_symbolic_links(tmp_path):
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(info, "../outside")
    with pytest.raises(ValueError, match="symbolic link"):
        safe_extract_zip(archive, tmp_path / "destination")


def test_safe_tar_preserves_executable_regular_files(tmp_path):
    archive = tmp_path / "safe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("bin/llama-server")
        info.mode = 0o755
        info.size = 4
        output.addfile(info, io.BytesIO(b"tool"))
    destination = tmp_path / "destination"
    safe_extract_tar(archive, destination)
    tool = destination / "bin" / "llama-server"
    assert tool.read_bytes() == b"tool"
    if os.name != "nt":
        assert tool.stat().st_mode & stat.S_IXUSR


def test_safe_tar_rejects_links_before_extracting(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        regular = tarfile.TarInfo("written-first")
        regular.size = 4
        output.addfile(regular, io.BytesIO(b"data"))
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        output.addfile(link)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="unsupported"):
        safe_extract_tar(archive, destination)
    assert not destination.exists()
