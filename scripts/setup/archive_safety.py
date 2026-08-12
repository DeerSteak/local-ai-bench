"""Path validation for downloaded setup archives."""

import re
import shutil
import stat
import tarfile
import zipfile
from posixpath import normpath
from pathlib import Path, PurePosixPath


DRIVE_PATH = re.compile(r"^[A-Za-z]:")


def validate_archive_names(names):
    """Reject member names that can escape or ambiguously address an extraction root."""
    validated = []
    for raw_name in names:
        if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
            raise ValueError("archive contains an empty or invalid member name")
        normalized = raw_name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if normalized.startswith("/") or DRIVE_PATH.match(normalized) or ".." in path.parts:
            raise ValueError(f"archive member escapes extraction root: {raw_name}")
        raw_parts = normalized.split("/")
        if raw_parts[-1] == "":
            raw_parts.pop()
        if not raw_parts or any(part in {"", "."} for part in raw_parts):
            raise ValueError(f"archive member has an ambiguous path: {raw_name}")
        validated.append(path.as_posix())
    if len(validated) != len(set(validated)):
        raise ValueError("archive contains duplicate normalized member paths")
    return tuple(validated)


def safe_extract_zip(archive_path, destination):
    """Extract regular ZIP members only after validating the complete manifest."""
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        validate_archive_names(member.filename for member in members)
        for member in members:
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"archive contains a symbolic link: {member.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / member.filename.replace("\\", "/")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _resolve_tar_link(name, members):
    seen = set()
    while True:
        if name in seen:
            raise ValueError(f"archive contains a symbolic link cycle: {name}")
        seen.add(name)
        member = members[name]
        target_name = normpath(
            (PurePosixPath(name).parent / member.linkname.replace("\\", "/")).as_posix()
        )
        validate_archive_names([target_name])
        target = members.get(target_name)
        if target is None or not (target.isfile() or target.issym()):
            raise ValueError(f"archive link does not target a regular member: {name}")
        if target.isfile():
            return target
        name = target_name


def safe_extract_tar(archive_path, destination):
    """Extract safe tar members, materializing in-archive file symlinks as copies."""
    destination = Path(destination)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = validate_archive_names(member.name for member in members)
        by_name = dict(zip(names, members))
        link_targets = {}
        for name, member in zip(names, members):
            if member.issym():
                link_targets[name] = _resolve_tar_link(name, by_name)
            elif not (member.isdir() or member.isfile()):
                raise ValueError(f"archive contains an unsupported member: {member.name}")
        destination.mkdir(parents=True, exist_ok=True)
        for name, member in zip(names, members):
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source_member = link_targets.get(name, member)
            source = archive.extractfile(source_member)
            if source is None:
                raise ValueError(f"archive member could not be read: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(source_member.mode & 0o777)


def validate_7z_archive(archive_path):
    """Validate 7z member paths before either native or Python extraction."""
    import py7zr
    with py7zr.SevenZipFile(str(archive_path), mode="r") as archive:
        return validate_archive_names(archive.getnames())
