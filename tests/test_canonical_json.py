import hashlib

import pytest

from scripts.results.canonical_json import canonical_json, canonical_json_bytes, sha256_json


def test_canonical_json_is_stable_across_mapping_order():
    first = {"b": [2, 1], "a": {"d": 4, "c": 3}}
    second = {"a": {"c": 3, "d": 4}, "b": [2, 1]}

    assert canonical_json(first) == '{"a":{"c":3,"d":4},"b":[2,1]}'
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_json(first) == hashlib.sha256(canonical_json_bytes(first)).hexdigest()


def test_canonical_json_rejects_nonfinite_numbers():
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})
