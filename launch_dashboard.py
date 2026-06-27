#!/usr/bin/env python3
"""
launch_dashboard.py — Build (if needed) and serve the benchmark dashboard.

Usage:
  python launch_dashboard.py             # http://localhost:3000
  python launch_dashboard.py --port 8080
  python launch_dashboard.py --rebuild   # force a fresh npm build
"""

import argparse
import http.server
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import webbrowser

SCRIPT_DIR    = pathlib.Path(__file__).resolve().parent
DASHBOARD_DIR = SCRIPT_DIR / "dashboard"
DIST_DIR      = DASHBOARD_DIR / "dist"
DEFAULT_PORT  = 3000


def npm_cmd():
    name = "npm.cmd" if sys.platform == "win32" else "npm"
    if shutil.which(name) is None:
        print("Error: npm not found in PATH.")
        print("Install Node.js from https://nodejs.org/ and re-run.")
        sys.exit(1)
    return name


def ensure_deps(npm):
    node_modules = DASHBOARD_DIR / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies (npm install) …")
        result = subprocess.run([npm, "install"], cwd=DASHBOARD_DIR)
        if result.returncode != 0:
            print("npm install failed — fix the errors above and try again.")
            sys.exit(1)
        print("Dependencies installed.\n")


def build(npm):
    print("Building dashboard …")
    result = subprocess.run([npm, "run", "build"], cwd=DASHBOARD_DIR)
    if result.returncode != 0:
        print("Build failed — fix the errors above and try again.")
        sys.exit(1)
    print("Build complete.\n")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve the benchmark dashboard")
    parser.add_argument("--port",    type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force a fresh npm build even if dist/ already exists")
    args = parser.parse_args()

    if not DASHBOARD_DIR.exists():
        print(f"Error: dashboard directory not found at {DASHBOARD_DIR}")
        sys.exit(1)

    npm = npm_cmd()
    ensure_deps(npm)

    needs_build = args.rebuild or not (DIST_DIR / "index.html").exists()
    if needs_build:
        build(npm)

    os.chdir(DIST_DIR)
    server = http.server.HTTPServer(("", args.port), _QuietHandler)
    url    = f"http://localhost:{args.port}"

    print(f"Dashboard → {url}")
    print("Drop your results JSON files onto the page to analyze them.")
    print("Ctrl-C to stop.\n")

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
