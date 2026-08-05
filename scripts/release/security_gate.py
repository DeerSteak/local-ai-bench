"""Offline secret and packaged-artifact checks for release staging trees."""

import re
from pathlib import Path


SCAN_CHUNK_BYTES = 1024 * 1024
SCAN_OVERLAP_BYTES = 512
PROHIBITED_NAMES = {
    ".env", ".git-credentials", ".npmrc", ".pypirc", "hf.txt", "id_rsa", "id_ed25519",
}
PROHIBITED_SUFFIXES = {".p12", ".pfx"}
SECRET_PATTERNS = (
    ("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[A-Z0-9]{16}\b")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer credential", re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}")),
    ("GitHub token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
    ("OpenAI-style API key", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GCP service account", re.compile(rb'"type"\s*:\s*"service_account"')),
    ("credential assignment", re.compile(rb"(?i)\b(?:api[_-]?key|password)\s*[=:]\s*[^\s\"']{8,}")),
)


def scan_file_patterns(path):
    matches = set()
    overlap = b""
    with Path(path).open("rb") as stream:
        while chunk := stream.read(SCAN_CHUNK_BYTES):
            content = overlap + chunk
            for label, pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    matches.add(label)
            overlap = content[-SCAN_OVERLAP_BYTES:]
    return matches


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
        if (candidate.name.lower() in PROHIBITED_NAMES
                or candidate.suffix.lower() in PROHIBITED_SUFFIXES):
            findings.append({"file": relative, "kind": "prohibited credential file", "blocking": True})
        try:
            matches = scan_file_patterns(candidate)
        except OSError:
            findings.append({"file": relative, "kind": "unreadable file", "blocking": True})
            continue
        for label in sorted(matches):
            findings.append({"file": relative, "kind": label, "blocking": True})
    return tuple(findings)


def security_gate_result(root):
    """Return a structured gate result without including matched secret values."""
    findings = scan_release_tree(root)
    return {"schema_version": 1, "passed": not findings, "findings": list(findings)}
