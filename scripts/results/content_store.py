"""Atomic content-addressed storage for large local artifacts."""

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactRef:
    sha256: str
    size: int
    media_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) != {"sha256", "size", "media_type"}:
            raise ValueError("artifact reference is incomplete")
        if (not isinstance(value["sha256"], str) or len(value["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in value["sha256"])
                or not isinstance(value["size"], int) or isinstance(value["size"], bool)
                or value["size"] < 0 or not isinstance(value["media_type"], str)
                or not value["media_type"]):
            raise ValueError("artifact reference is invalid")
        return cls(**value)


class ContentStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, reference: ArtifactRef) -> Path:
        reference = ArtifactRef.from_dict(reference.to_dict())
        return self.root / reference.sha256[:2] / reference.sha256[2:]

    def put_bytes(self, data: bytes, media_type: str) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise ValueError("artifact data must be bytes")
        digest = hashlib.sha256(data).hexdigest()
        reference = ArtifactRef(digest, len(data), media_type)
        self._store(reference, [data])
        return reference

    def put_file(self, source: Path, media_type: str) -> ArtifactRef:
        digest = hashlib.sha256()
        size = 0
        descriptor, temporary_name = tempfile.mkstemp(dir=self.root, prefix=".artifact.", suffix=".tmp")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as target, Path(source).open("rb") as source_stream:
                for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        reference = ArtifactRef(digest.hexdigest(), size, media_type)
        self._commit_temporary(reference, temporary)
        return reference

    def _store(self, reference: ArtifactRef, chunks) -> None:
        destination = self.path_for(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self.verify(reference)
            return
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=".artifact.", suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                for chunk in chunks:
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            self._commit_temporary(reference, temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def _commit_temporary(self, reference: ArtifactRef, temporary: Path) -> None:
        destination = self.path_for(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self.verify(reference)
            temporary.unlink(missing_ok=True)
            return
        os.replace(temporary, destination)
        if os.name != "nt":
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def verify(self, reference: ArtifactRef) -> Path:
        path = self.path_for(reference)
        if not path.is_file():
            raise ValueError(f"artifact is missing: {reference.sha256}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != reference.sha256 or size != reference.size:
            raise ValueError(f"artifact integrity check failed: {reference.sha256}")
        return path

    def read_bytes(self, reference: ArtifactRef, max_bytes: int | None = None) -> bytes:
        path = self.verify(reference)
        if max_bytes is not None and reference.size > max_bytes:
            raise ValueError(f"artifact exceeds read limit: {reference.size} bytes")
        return path.read_bytes()
