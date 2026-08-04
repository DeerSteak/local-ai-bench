"""Deterministic release checksums and provenance manifest."""

import hashlib
import json
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1


def build_release_manifest(repo_root, artifacts, *, application_version, channel, source_commit):
    """Hash explicit regular files beneath the repository release root."""
    root = Path(repo_root).resolve()
    if channel not in {"preview", "stable"}:
        raise ValueError("release channel must be preview or stable")
    if not application_version or not source_commit:
        raise ValueError("release provenance requires version and source commit")
    files = {}
    for artifact in artifacts:
        supplied = Path(artifact)
        if supplied.is_symlink():
            raise ValueError(f"release artifact must be a regular file: {supplied}")
        path = supplied.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"release artifact is outside the release root: {path}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release artifact must be a regular file: {path}")
        name = relative.as_posix()
        if name in files:
            raise ValueError(f"duplicate release artifact: {name}")
        files[name] = _file_record(path)
    if not files:
        raise ValueError("release manifest requires at least one artifact")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "application_version": str(application_version),
        "channel": channel,
        "source_commit": source_commit,
        "files": dict(sorted(files.items())),
        "signature": {"status": "unsigned", "scheme": None, "value": None},
    }


def verify_release_manifest(repo_root, manifest):
    """Verify every declared file and reject malformed or unsigned stable releases."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False, ("unsupported manifest schema",)
    root = Path(repo_root).resolve()
    errors = []
    for name, expected in sorted(manifest.get("files", {}).items()):
        supplied = root / name
        if supplied.is_symlink():
            errors.append(f"missing artifact: {name}")
            continue
        path = supplied.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"unsafe artifact path: {name}")
            continue
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
        elif _file_record(path) != expected:
            errors.append(f"artifact digest mismatch: {name}")
    if manifest.get("channel") == "stable" and manifest.get("signature", {}).get("status") != "verified":
        errors.append("stable manifest signature is not verified")
    return not errors, tuple(errors)


def write_release_manifest(path, manifest):
    """Write canonical JSON suitable for detached signing."""
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}
