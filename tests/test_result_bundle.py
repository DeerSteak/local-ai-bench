import json
import zipfile
from pathlib import Path

import pytest

from scripts.results.result_bundle import (
    aggregate_reproduction_errors, export_result_bundle, import_result_bundle,
    methodology_availability_errors, verify_result_bundle,
)
from scripts.results.llm_event_stage import event_store_path
from scripts.results.local_execution_context import local_execution_path


FIXTURE = Path(__file__).parent / "fixtures" / "results_v4_1_complete.json"
SAMPLE_BUNDLE = Path(__file__).parents[1] / "samples" / "representative_v4_1.labresult"


def test_bundle_export_is_deterministic_and_imports_content_addressed_artifacts(tmp_path):
    artifact = tmp_path / "answers.json"
    artifact.write_text('{"answer": "B"}', encoding="utf-8")
    first = tmp_path / "first.labresult"
    second = tmp_path / "second.labresult"
    export_result_bundle(FIXTURE, first, [artifact])
    export_result_bundle(FIXTURE, second, [artifact])
    assert first.read_bytes() == second.read_bytes()
    verified = verify_result_bundle(first)
    assert verified["result"]["version"] == "4.1"

    imported = tmp_path / "imported.json"
    artifacts = tmp_path / "artifacts"
    manifest = import_result_bundle(first, imported, artifacts)
    assert json.loads(imported.read_text()) == verified["result"]
    extracted = list(artifacts.iterdir())
    assert len(extracted) == 1
    assert extracted[0].name.startswith(manifest["artifacts"][0]["sha256"])
    assert extracted[0].read_bytes() == artifact.read_bytes()


def test_representative_onboarding_bundle_is_current_and_clearly_synthetic():
    verified = verify_result_bundle(SAMPLE_BUNDLE)
    assert verified["manifest"]["application_version"] == "4.1"
    assert verified["manifest"]["result_schema_version"] == 3
    profile = verified["result"]["profile"]
    assert profile["hostname"] == "Synthetic Sample System"
    assert profile["gpu"] == "Synthetic Sample Hardware"


def test_bundle_export_applies_private_aliases_and_retains_source_identity(tmp_path):
    bundle = tmp_path / "aliased.labresult"
    export_result_bundle(
        FIXTURE, bundle, system_alias="System A", hardware_alias="Hardware A",
    )
    exported = verify_result_bundle(bundle)["result"]
    assert exported["profile"]["hostname"] == "System A"
    assert exported["profile"]["gpu"] == "Hardware A"
    identity = exported["run"]["export_identity"]
    assert identity["aliases_applied"] == ["system", "hardware"]
    assert len(identity["source_sha256"]) == 64
    assert json.loads(FIXTURE.read_text())["profile"]["hostname"] == "commercial-golden-system"


def test_bundle_export_refuses_private_local_execution_context(tmp_path):
    result = tmp_path / "results_private.json"
    result.write_bytes(FIXTURE.read_bytes())
    private = local_execution_path(event_store_path(result))
    private.write_text('{"comfyui_dir":"/private/path"}', encoding="utf-8")
    with pytest.raises(ValueError, match="private local execution context"):
        export_result_bundle(result, tmp_path / "private.labresult", [private])


def test_bundle_verifier_rejects_tampered_payload(tmp_path):
    bundle = tmp_path / "result.labresult"
    export_result_bundle(FIXTURE, bundle)
    tampered = tmp_path / "tampered.labresult"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "result.json":
                data += b" "
            target.writestr(info, data)
    with pytest.raises(ValueError, match="integrity check failed"):
        verify_result_bundle(tampered)


def test_bundle_verifier_rejects_path_bearing_artifact_name(tmp_path):
    artifact = tmp_path / "answers.json"
    artifact.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "result.labresult"
    export_result_bundle(FIXTURE, bundle, [artifact])
    rewritten = tmp_path / "unsafe.labresult"
    with zipfile.ZipFile(bundle) as source:
        manifest = json.loads(source.read("manifest.json"))
        manifest["artifacts"][0]["original_name"] = "../answers.json"
        with zipfile.ZipFile(rewritten, "w") as target:
            for info in source.infolist():
                data = (json.dumps(manifest).encode() if info.filename == "manifest.json"
                        else source.read(info.filename))
                target.writestr(info, data)
    with pytest.raises(ValueError, match="artifact inventory"):
        verify_result_bundle(rewritten)


def test_aggregate_reproduction_detects_modified_mean():
    result = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result["llm"]["golden"]["2K"]["tps_mean"] = 999
    assert aggregate_reproduction_errors(result) == [
        "$.llm.golden.2K.tps_mean does not match valid_samples",
    ]


def test_methodology_availability_checks_local_bank_identity():
    result = {"bank_versions": {"mcq": "wrong", "unknown": "hash"}}
    assert methodology_availability_errors(result) == [
        "Methodology bank version does not match: mcq",
        "Methodology bank is unavailable: unknown",
    ]
