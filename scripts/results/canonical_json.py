"""Canonical JSON encoding and hashing for durable result identities."""

import hashlib
import json


def canonical_json(value) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def canonical_json_bytes(value) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()
