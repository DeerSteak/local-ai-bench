import copy

import pytest

from outbound_metadata import (
    format_outbound_preview, outbound_metadata_preview, prepare_outbound_result,
    source_identity_digest, verify_source_identity,
)


def result():
    return {
        "version": "4.1", "engine": "llamacpp",
        "profile": {"hostname": "Secret Prototype", "gpu": "Unreleased GPU", "ram_gb": 128},
        "run": {"plan_id": "abc", "models": {"llm": [{"tag": "vendor:model"}]}},
    }


def test_preview_lists_outbound_identity_and_model_metadata():
    preview = dict(outbound_metadata_preview(result()))
    assert preview["profile.hostname"] == "Secret Prototype"
    assert preview["profile.gpu"] == "Unreleased GPU"
    assert preview["models.llm[0]"] == "vendor:model"
    assert "profile.hostname: Secret Prototype" in format_outbound_preview(result())


def test_aliases_replace_exported_names_and_preserve_source_with_digest():
    source = result()
    original = copy.deepcopy(source)
    outbound = prepare_outbound_result(
        source, system_alias="System A", hardware_alias="Hardware A",
    )
    assert source == original
    assert outbound["profile"]["hostname"] == "System A"
    assert outbound["profile"]["gpu"] == "Hardware A"
    assert outbound["run"]["export_identity"] == {
        "source_sha256": source_identity_digest(source),
        "aliases_applied": ["system", "hardware"],
    }
    assert source_identity_digest(source) == source_identity_digest(copy.deepcopy(source))
    assert verify_source_identity(outbound, source)
    different = result()
    different["profile"]["hostname"] = "Other"
    assert not verify_source_identity(outbound, different)


def test_unaliased_export_still_records_verifiable_source_identity():
    outbound = prepare_outbound_result(result())
    assert outbound["run"]["export_identity"]["aliases_applied"] == []


@pytest.mark.parametrize(("field", "value"), [("system_alias", ""), ("hardware_alias", "  ")])
def test_aliases_reject_empty_values(field, value):
    with pytest.raises(ValueError, match="alias"):
        prepare_outbound_result(result(), **{field: value})
