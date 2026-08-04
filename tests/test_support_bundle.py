import json
import zipfile

from support_bundle import build_support_payload, export_support_bundle, preview_support_bundle


def sensitive_result():
    return {
        "version": "4.1", "engine": "llamacpp",
        "profile": {
            "hostname": "secret-prelaunch-machine", "os": "Test OS", "arch": "arm64",
            "ram_gb": 64, "backend": "metal",
        },
        "run": {
            "schema_version": 3, "status": "failed", "reason": "stage_execution_failed",
            "requested_tests": ["llm"], "plan_id": "abc", "source": {"git_commit": "deadbeef"},
            "stages": {"llm": {"status": "failed", "selected_models": 1,
                                  "models_with_results": 1, "models_failed": 0}},
        },
        "llm": {"private-model": {
            "label": "Unreleased Model", "prompt": "secret prompt", "response": "secret response",
            "2K": {"tokens_per_sec": 50},
            "error": "failed reading /Users/alice/private/model.gguf with hf_abcdefghijklmnop",
        }},
    }


def test_support_payload_is_allowlisted_and_redacts_diagnostics():
    payload = build_support_payload(sensitive_result())
    encoded = json.dumps(payload)
    assert payload["system"] == {"os": "Test OS", "arch": "arm64", "ram_gb": 64, "backend": "metal"}
    for secret in ("secret-prelaunch-machine", "private-model", "secret prompt", "secret response",
                   "/Users/alice", "hf_abcdefghijklmnop"):
        assert secret not in encoded
    assert "<private-path>" in encoded and "<secret>" in encoded
    assert payload["run"]["stages"]["llm"]["models_with_results"] == 1


def test_support_preview_lists_every_exported_field_and_file(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(sensitive_result()), encoding="utf-8")
    preview = preview_support_bundle(result)
    assert preview["files"] == ["support.json", "manifest.json"]
    assert "$.application.version" in preview["fields"]
    assert any(field.endswith(".details.error") for field in preview["fields"])


def test_support_export_is_deterministic_and_contains_only_previewed_files(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(sensitive_result()), encoding="utf-8")
    first = tmp_path / "first.labsupport"
    second = tmp_path / "second.labsupport"
    export_support_bundle(result, first)
    export_support_bundle(result, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert sorted(archive.namelist()) == ["manifest.json", "support.json"]
        encoded = archive.read("support.json").decode()
    assert "secret prompt" not in encoded
