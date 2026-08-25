import json

import pytest

from scripts.app.workspace_server import (
    build_workspace_export, evaluate_workspace, workspace_request_authorized,
)
from scripts.results.workspace_export import verify_workspace_bundle
from scripts.results.workspace_selection import build_workspace_selection


FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "results_v4_1_complete.json"


def payload(tmp_path, output_format="html"):
    result = tmp_path / "result.json"
    result.write_bytes(FIXTURE.read_bytes())
    return {
        "format": output_format,
        "selection": build_workspace_selection([result]),
        "results": [{"name": result.name, "text": result.read_text(encoding="utf-8")}],
    }


@pytest.mark.parametrize(("output_format", "content_type", "prefix"), [
    ("html", "text/html; charset=utf-8", b"<!doctype html>"),
    ("pdf", "application/pdf", b"%PDF-"),
    ("bundle", "application/zip", b"PK"),
])
def test_workspace_server_builds_bounded_exports(tmp_path, output_format, content_type, prefix):
    data, actual_type, filename = build_workspace_export(payload(tmp_path, output_format))
    assert data.startswith(prefix)
    assert actual_type == content_type
    assert filename
    if output_format == "bundle":
        path = tmp_path / filename
        path.write_bytes(data)
        assert verify_workspace_bundle(path)["selection"]["artifact_type"] == "workspace_selection"


def test_workspace_server_rejects_unknown_shape_format_and_changed_content(tmp_path):
    request = payload(tmp_path)
    with pytest.raises(ValueError, match="invalid"):
        build_workspace_export({**request, "path": "/private"})
    with pytest.raises(ValueError, match="format"):
        build_workspace_export({**request, "format": "command"})
    request["results"][0]["text"] += " "
    with pytest.raises(ValueError, match="missing or changed"):
        build_workspace_export(request)


def test_workspace_evaluation_applies_embedded_policy_to_recorded_baseline(tmp_path):
    request = payload(tmp_path)
    request["selection"]["acceptance_policy"] = {
        "schema_version": 1, "name": "Gate", "methodology_profile": "neutral-v1",
        "rules": [{
            "id": "throughput", "section": "llm", "model": "golden", "case": "2K",
            "metric": "tps_mean", "operator": "at_least", "threshold": 55,
            "minimum_evidence": 2,
        }],
    }
    assert evaluate_workspace({
        "selection": request["selection"], "results": request["results"],
    })["acceptance"]["decision"] == "rejected"


@pytest.mark.parametrize(("host", "origin", "authorization", "allowed"), [
    ("127.0.0.1:3000", "http://127.0.0.1:3000", "Bearer secret", True),
    ("localhost:3000", "http://localhost:3000", "Bearer secret", True),
    ("evil.test", "http://127.0.0.1:3000", "Bearer secret", False),
    ("127.0.0.1:3000", "https://evil.test", "Bearer secret", False),
    ("127.0.0.1:3000", "http://127.0.0.1:3000", "Bearer wrong", False),
])
def test_workspace_http_boundary_enforces_host_origin_and_token(
        host, origin, authorization, allowed):
    assert workspace_request_authorized(host, origin, authorization, "secret", 3000) is allowed
