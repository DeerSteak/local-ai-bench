"""Offline secret and packaged-artifact checks for release staging trees."""

import re
from pathlib import Path


MAX_SCANNED_FILE_BYTES = 10 * 1024 * 1024
PROHIBITED_NAMES = {".env", "hf.txt", "id_rsa", "id_ed25519"}
SECRET_PATTERNS = (
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[A-Z0-9]{16}\b")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer credential", re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}")),
)


def scan_release_tree(root):
    """Return deterministic findings for regular files in a release staging tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("security scan requires a release staging directory")
    findings = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            findings.append({"file": relative, "kind": "symbolic link", "blocking": True})
            continue
        if not candidate.is_file():
            continue
        if candidate.name.lower() in PROHIBITED_NAMES:
            findings.append({"file": relative, "kind": "prohibited credential file", "blocking": True})
        size = candidate.stat().st_size
        if size > MAX_SCANNED_FILE_BYTES:
            findings.append({"file": relative, "kind": "file exceeds offline scan limit", "blocking": True})
            continue
        try:
            content = candidate.read_bytes()
        except OSError:
            findings.append({"file": relative, "kind": "unreadable file", "blocking": True})
            continue
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append({"file": relative, "kind": label, "blocking": True})
    return tuple(findings)


def security_gate_result(root):
    """Return a structured gate result without including matched secret values."""
    findings = scan_release_tree(root)
    return {"schema_version": 1, "passed": not findings, "findings": list(findings)}
