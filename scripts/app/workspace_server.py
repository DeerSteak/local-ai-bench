"""Loopback-only static workspace server with bounded artifact exports."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import tempfile
from urllib.parse import urlparse
import webbrowser

from scripts.results.workspace_export import export_workspace_bundle, write_workspace_reports
from scripts.results.acceptance_policy import evaluate_policy
from scripts.results.recommendation import validate_recommendation_artifact
from scripts.results.workspace_export import resolve_workspace_results
from scripts.results.workspace_selection import validate_workspace_selection


MAX_EXPORT_REQUEST_BYTES = 256 * 1024 * 1024
EXPORT_FORMATS = {
    "bundle": ("application/zip", "workspace.labworkspace"),
    "html": ("text/html; charset=utf-8", "decision.html"),
    "pdf": ("application/pdf", "decision.pdf"),
}


def workspace_request_authorized(host: str | None, origin: str | None,
                                 authorization: str | None, token: str, port: int) -> bool:
    return host in {f"127.0.0.1:{port}", f"localhost:{port}"} \
        and origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"} \
        and authorization == f"Bearer {token}"


def build_workspace_export(payload: dict) -> tuple[bytes, str, str]:
    if not isinstance(payload, dict) or set(payload) != {"format", "selection", "results"}:
        raise ValueError("workspace export request is invalid")
    output_format = payload.get("format")
    if output_format not in EXPORT_FORMATS:
        raise ValueError("unsupported workspace export format")
    selection_value = payload.get("selection")
    if not isinstance(selection_value, dict):
        raise ValueError("workspace export selection is invalid")
    selection = validate_workspace_selection(selection_value)
    sources = payload.get("results")
    if not isinstance(sources, list) or len(sources) != len(selection["results"]):
        raise ValueError("workspace export result inventory is invalid")
    with tempfile.TemporaryDirectory(prefix="local-ai-bench-workspace-") as directory:
        root = Path(directory)
        candidates = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict) or set(source) != {"name", "text"} \
                    or not isinstance(source["name"], str) or not isinstance(source["text"], str):
                raise ValueError("workspace export result source is invalid")
            path = root / f"{index}.json"
            path.write_text(source["text"], encoding="utf-8", newline="")
            candidates.append(path)
        content_type, filename = EXPORT_FORMATS[output_format]
        output = root / filename
        if output_format == "bundle":
            export_workspace_bundle(selection, candidates, output)
        else:
            write_workspace_reports(
                selection, candidates,
                html_path=output if output_format == "html" else None,
                pdf_path=output if output_format == "pdf" else None,
            )
        return output.read_bytes(), content_type, filename


def evaluate_workspace(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"selection", "results"}:
        raise ValueError("workspace evaluation request is invalid")
    selection_value = payload.get("selection")
    if not isinstance(selection_value, dict):
        raise ValueError("workspace evaluation selection is invalid")
    selection = validate_workspace_selection(selection_value)
    sources = payload.get("results")
    if not isinstance(sources, list) or len(sources) != len(selection["results"]):
        raise ValueError("workspace evaluation result inventory is invalid")
    with tempfile.TemporaryDirectory(prefix="local-ai-bench-workspace-") as directory:
        root = Path(directory)
        candidates = []
        for index, source in enumerate(sources):
            if not isinstance(source, dict) or set(source) != {"name", "text"} \
                    or not isinstance(source["text"], str):
                raise ValueError("workspace evaluation result source is invalid")
            path = root / f"{index}.json"
            path.write_text(source["text"], encoding="utf-8", newline="")
            candidates.append(path)
        paths = resolve_workspace_results(selection, candidates)
        baseline = selection["baseline_sha256"] or selection["results"][0]["sha256"]
        selected_index = next(index for index, item in enumerate(selection["results"])
                              if item["sha256"] == baseline)
        result = json.loads(paths[selected_index].read_text(encoding="utf-8"))
        policy = selection.get("acceptance_policy")
        recommendation = selection.get("recommendation")
        if recommendation is not None:
            validate_recommendation_artifact(recommendation, source_result=result)
        return {
            "acceptance": evaluate_policy(result, policy) if policy is not None else None,
            "recommendation": recommendation,
        }


def workspace_handler(dist_directory: Path, token: str, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class WorkspaceHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dist_directory), **kwargs)

        def _same_origin(self) -> bool:
            return workspace_request_authorized(
                self.headers.get("Host"), self.headers.get("Origin"),
                self.headers.get("Authorization"), token, port,
            )

        def do_GET(self):
            if self.headers.get("Host") not in allowed_hosts:
                self.send_error(403)
                return
            if urlparse(self.path).path == "/__workspace_config__.json":
                data = json.dumps({"token": token}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            super().do_GET()

        def do_POST(self):
            request_path = urlparse(self.path).path
            if request_path not in {"/api/workspace/export", "/api/workspace/evaluate"} \
                    or not self._same_origin():
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_EXPORT_REQUEST_BYTES:
                    raise ValueError("workspace export request size is invalid")
                payload = json.loads(self.rfile.read(length))
                if request_path == "/api/workspace/evaluate":
                    data = json.dumps(evaluate_workspace(payload)).encode("utf-8")
                    content_type, filename = "application/json", None
                else:
                    data, content_type, filename = build_workspace_export(payload)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                message = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            if filename is not None:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return WorkspaceHandler


def main(argv=None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Local AI Bench decision workspace")
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--open-path", default="/")
    args = parser.parse_args(argv)
    if not args.dist.is_dir() or not 1 <= args.port <= 65535:
        parser.error("a built dashboard directory and valid port are required")
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), workspace_handler(args.dist, token, args.port),
    )
    url = f"http://127.0.0.1:{args.port}{args.open_path}"
    print(f"Dashboard -> {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
