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


@pytest.mark.parametrize("terminal_error", [OSError(5, "terminal revoked"), BrokenPipeError()])
@pytest.mark.parametrize("request_path", ["/__workspace_config__.json", "/index.html", "/missing"])
def test_http_responses_survive_disconnected_terminal(
        tmp_path, monkeypatch, terminal_error, request_path):
    import io
    from scripts.app.workspace_server import workspace_handler

    class BrokenTerminal:
        def write(self, _message):
            raise terminal_error

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *_args):
            return io.BytesIO(
                f"GET {request_path} HTTP/1.1\r\nHost: 127.0.0.1:3000\r\n\r\n".encode(),
            )

        def sendall(self, data):
            self.output.extend(data)

    (tmp_path / "index.html").write_text("dashboard content")
    connection = Connection()
    monkeypatch.setattr("sys.stderr", BrokenTerminal())
    workspace_handler(tmp_path, "test-token", 3000)(
        connection, ("127.0.0.1", 1234), object(),
    )
    headers, body = bytes(connection.output).split(b"\r\n\r\n", 1)
    if request_path == "/missing":
        assert headers.startswith(b"HTTP/1.0 404")
    else:
        assert headers.startswith(b"HTTP/1.0 200")
        if request_path == "/__workspace_config__.json":
            assert json.loads(body) == {"token": "test-token"}
        else:
            assert body == b"dashboard content"


def test_bind_workspace_server_returns_bound_server(tmp_path, monkeypatch):
    from scripts.app import workspace_server

    server = object()
    addresses = []

    def bind(address, handler):
        addresses.append(address)
        return server

    monkeypatch.setattr(workspace_server, "ThreadingHTTPServer", bind)
    assert workspace_server.bind_workspace_server(tmp_path, 4321, "/") is server
    assert addresses == [("127.0.0.1", 4321)]


@pytest.mark.parametrize("reusable", [True, False])
def test_bind_workspace_server_handles_port_taken_after_launcher_probe(
        tmp_path, monkeypatch, reusable):
    import errno
    from scripts.app import workspace_server

    def occupied(*_args):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    reopened = []

    def reopen(port, path):
        reopened.append((port, path))
        return reusable

    monkeypatch.setattr(workspace_server, "ThreadingHTTPServer", occupied)
    monkeypatch.setattr(workspace_server, "reopen_dashboard", reopen)
    if reusable:
        assert workspace_server.bind_workspace_server(tmp_path, 4321, "/?autoload=1") is None
    else:
        with pytest.raises(SystemExit, match="port 4321 is already in use.*--port"):
            workspace_server.bind_workspace_server(tmp_path, 4321, "/?autoload=1")
    assert reopened == [(4321, "/?autoload=1")]


def test_bind_workspace_server_preserves_other_os_errors(tmp_path, monkeypatch):
    import errno
    from scripts.app import workspace_server

    error = OSError(errno.EACCES, "Permission denied")

    def denied(*_args):
        raise error

    monkeypatch.setattr(workspace_server, "ThreadingHTTPServer", denied)
    with pytest.raises(OSError) as caught:
        workspace_server.bind_workspace_server(tmp_path, 4321, "/")
    assert caught.value is error
