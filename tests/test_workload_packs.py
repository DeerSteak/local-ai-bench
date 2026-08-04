import json

import pytest

from workload_packs import BUILTIN_PACKS, load_custom_pack, validate_pack


def test_builtin_packs_are_valid_and_have_repeatable_identity():
    first = validate_pack(BUILTIN_PACKS["core-v1"])
    second = validate_pack(BUILTIN_PACKS["core-v1"])
    assert first["digest"] == second["digest"]
    assert first["stages"] == BUILTIN_PACKS["core-v1"]["stages"]


@pytest.mark.parametrize("change, message", [
    ({"schema_version": 2}, "schema"),
    ({"version": 0}, "positive integer"),
    ({"stages": ["llm", "llm"]}, "unique"),
    ({"stages": ["future"]}, "unsupported"),
    ({"origin": "remote"}, "origin"),
])
def test_invalid_pack_contract_is_rejected(change, message):
    pack = dict(BUILTIN_PACKS["core-v1"])
    pack.update(change)
    with pytest.raises(ValueError, match=message):
        validate_pack(pack)


def test_incompatible_application_version_is_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        validate_pack(BUILTIN_PACKS["core-v1"], "4.2")


def test_unknown_or_missing_fields_are_rejected():
    pack = dict(BUILTIN_PACKS["core-v1"])
    del pack["id"]
    pack["script"] = "do-not-run.py"
    with pytest.raises(ValueError, match="invalid workload-pack fields"):
        validate_pack(pack)


def test_custom_pack_loads_as_data_only(tmp_path):
    path = tmp_path / "focused.labpack.json"
    path.write_text(json.dumps({
        "schema_version": 1, "id": "focused", "version": 1,
        "stages": ["llm", "conv"], "application_versions": ["4.1"],
        "origin": "custom",
    }), encoding="utf-8")
    loaded = load_custom_pack(path)
    assert loaded["id"] == "focused"
    assert loaded["digest"].startswith("sha256:")


def test_custom_loader_rejects_builtin_origin(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(BUILTIN_PACKS["core-v1"]), encoding="utf-8")
    with pytest.raises(ValueError, match="custom origin"):
        load_custom_pack(path)
