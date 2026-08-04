import os

import pytest

from scripts.results import content_store
from scripts.results.content_store import ArtifactRef, ContentStore


def test_content_store_deduplicates_bytes_and_uses_digest_only_paths(tmp_path):
    store = ContentStore(tmp_path / "objects")
    first = store.put_bytes(b"same content", "text/plain")
    second = store.put_bytes(b"same content", "text/plain")
    assert first == second
    assert store.path_for(first).relative_to(store.root).parts == (
        first.sha256[:2], first.sha256[2:],
    )
    assert store.read_bytes(first) == b"same content"
    assert len(list(store.root.rglob(first.sha256[2:]))) == 1


def test_put_file_streams_and_matches_byte_identity(tmp_path):
    source = tmp_path / "private customer filename.log"
    source.write_bytes((b"large-ish-content" * 100000) + b"end")
    store = ContentStore(tmp_path / "objects")
    from_file = store.put_file(source, "text/plain")
    from_bytes = store.put_bytes(source.read_bytes(), "text/plain")
    assert from_file == from_bytes
    assert source.name not in str(store.path_for(from_file))


def test_content_store_detects_corruption_and_missing_content(tmp_path):
    store = ContentStore(tmp_path / "objects")
    reference = store.put_bytes(b"original", "application/octet-stream")
    store.path_for(reference).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity check failed"):
        store.verify(reference)
    missing = ArtifactRef("0" * 64, 1, "application/octet-stream")
    with pytest.raises(ValueError, match="missing"):
        store.verify(missing)


def test_read_limit_is_checked_before_loading_content(tmp_path, monkeypatch):
    store = ContentStore(tmp_path / "objects")
    reference = store.put_bytes(b"12345", "text/plain")
    monkeypatch.setattr(content_store.Path, "read_bytes", lambda *_: (_ for _ in ()).throw(
        AssertionError("must not read")))
    with pytest.raises(ValueError, match="exceeds read limit"):
        store.read_bytes(reference, max_bytes=4)


@pytest.mark.parametrize("value", [
    {}, {"sha256": "x", "size": 1, "media_type": "text/plain"},
    {"sha256": "0" * 64, "size": -1, "media_type": "text/plain"},
    {"sha256": "0" * 64, "size": 1, "media_type": ""},
])
def test_artifact_reference_rejects_malformed_values(value):
    with pytest.raises(ValueError):
        ArtifactRef.from_dict(value)


def test_failed_replace_leaves_no_partial_object(tmp_path, monkeypatch):
    store = ContentStore(tmp_path / "objects")
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        store.put_bytes(b"content", "text/plain")
    assert not list(store.root.rglob(".artifact.*.tmp"))
