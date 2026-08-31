"""Detect and reopen an existing Local AI Bench dashboard server."""

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser


def dashboard_server_running(port: int, *, open_url=urlopen) -> bool:
    try:
        with open_url(
                f"http://127.0.0.1:{port}/__workspace_config__.json", timeout=0.5) as response:
            payload = json.load(response)
            return response.status == 200 and isinstance(payload, dict) \
                and isinstance(payload.get("token"), str) \
                and bool(payload["token"])
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def reopen_dashboard(port: int, open_path: str, *, open_url=urlopen,
                     open_browser=webbrowser.open) -> bool:
    if not dashboard_server_running(port, open_url=open_url):
        return False
    open_browser(f"http://127.0.0.1:{port}{open_path}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reuse a running Local AI Bench dashboard")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--open-path", required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not args.open_path.startswith("/"):
        parser.error("a valid port and root-relative open path are required")
    return 0 if reopen_dashboard(args.port, args.open_path) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
