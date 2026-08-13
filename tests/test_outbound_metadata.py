import copy

import pytest

from scripts.results.outbound_metadata import (
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


def test_outbound_telemetry_omits_raw_and_identity_fields():
    source = result()
    source["llm"] = {"model": {"2K": {"memory": {
        "case_id": "case-1",
        "windows": [{
            "name": "measured", "sample_count": 1, "duration_sec": 1,
            "channels": {"process_rss_gb": {
                "peak_gb": 4, "mean_gb": 4, "final_gb": 4, "valid_samples": 1,
                "device_uuid": "GPU-secret",
            }},
            "raw_output": "serial=secret path=/private/model.gguf",
        }],
        "summary": {"process_rss_gb": {"peak_gb": 4, "arguments": "--secret"}},
        "headroom": {"absolute_gb": 8, "fraction": 0.5, "state": "comfortable"},
        "provenance": {
            "interval_sec": 1, "failed_samples": 0,
            "channels": {"process_rss_gb": {
                "source": "psutil", "failed_samples": 2, "serial_number": "secret",
            }},
            "private_path": "/private/model.gguf",
        },
        "command_output": "forbidden",
    }}}}
    memory = prepare_outbound_result(source)["llm"]["model"]["2K"]["memory"]
    serialized = str(memory)
    assert memory["case_id"] == "case-1"
    assert memory["provenance"]["channels"]["process_rss_gb"] == {
        "source": "psutil", "failed_samples": 2,
    }
    for forbidden in ("secret", "private", "arguments", "raw_output", "command_output"):
        assert forbidden not in serialized.lower()


@pytest.mark.parametrize(("field", "value"), [("system_alias", ""), ("hardware_alias", "  ")])
def test_aliases_reject_empty_values(field, value):
    with pytest.raises(ValueError, match="alias"):
        prepare_outbound_result(result(), **{field: value})
