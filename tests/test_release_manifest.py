import json

import pytest

from scripts.release.release_manifest import build_release_manifest, verify_release_manifest, write_release_manifest


def test_manifest_is_deterministic_and_records_provenance(tmp_path):
    first = tmp_path / "app.zip"
    second = tmp_path / "sbom.json"
    first.write_bytes(b"application")
    second.write_bytes(b"sbom")
    one = build_release_manifest(
        tmp_path, [second, first], application_version="4.1", channel="preview", source_commit="abc123",
    )
    two = build_release_manifest(
        tmp_path, [first, second], application_version="4.1", channel="preview", source_commit="abc123",
    )
    assert one == two
    assert list(one["files"]) == ["app.zip", "sbom.json"]
    assert one["signature"]["status"] == "unsigned"


def test_manifest_rejects_outside_duplicate_missing_and_invalid_channel(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.write_text("x", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside-release-artifact"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        build_release_manifest(tmp_path, [outside], application_version="4.1", channel="preview", source_commit="a")
    with pytest.raises(ValueError, match="duplicate"):
        build_release_manifest(tmp_path, [artifact, artifact], application_version="4.1", channel="preview", source_commit="a")
    with pytest.raises(ValueError, match="at least one"):
        build_release_manifest(tmp_path, [], application_version="4.1", channel="preview", source_commit="a")
    with pytest.raises(ValueError, match="channel"):
        build_release_manifest(tmp_path, [artifact], application_version="4.1", channel="nightly", source_commit="a")


def test_manifest_rejects_symlinked_artifact(tmp_path, symlink_or_skip):
    target = tmp_path / "target"
    link = tmp_path / "link"
    target.write_text("x", encoding="utf-8")
    symlink_or_skip(link, target)
    with pytest.raises(ValueError, match="regular file"):
        build_release_manifest(tmp_path, [link], application_version="4.1", channel="preview", source_commit="a")


def test_verifier_detects_tampering_and_requires_stable_signature(tmp_path):
    artifact = tmp_path / "app.zip"
    artifact.write_bytes(b"original")
    preview = build_release_manifest(
        tmp_path, [artifact], application_version="4.1", channel="preview", source_commit="abc",
    )
    assert verify_release_manifest(tmp_path, preview) == (True, ())
    artifact.write_bytes(b"changed")
    assert verify_release_manifest(tmp_path, preview)[1] == ("artifact digest mismatch: app.zip",)
    artifact.write_bytes(b"original")
    stable = dict(preview, channel="stable")
    assert verify_release_manifest(tmp_path, stable)[1] == ("stable manifest signature is not verified",)


def test_canonical_manifest_output_round_trips(tmp_path):
    artifact = tmp_path / "app.zip"
    artifact.write_bytes(b"app")
    manifest = build_release_manifest(
        tmp_path, [artifact], application_version="4.1", channel="preview", source_commit="abc",
    )
    output = tmp_path / "release.json"
    write_release_manifest(output, manifest)
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
