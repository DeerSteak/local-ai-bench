#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$SCRIPT_DIR/dashboard"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT=3000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ! -d "$DASHBOARD_DIR" ]; then
    echo "Error: dashboard directory not found at $DASHBOARD_DIR"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm not found in PATH."
    echo "Install Node.js from https://nodejs.org/ and re-run."
    exit 1
fi

if [ ! -d "$DASHBOARD_DIR/node_modules" ]; then
    echo "Installing dependencies (npm install) ..."
    (cd "$DASHBOARD_DIR" && npm install)
    echo "Dependencies installed."
    echo
fi

echo "Building dashboard ..."
(cd "$DASHBOARD_DIR" && npm run build)
echo "Build complete."
echo

echo "Dashboard -> http://localhost:$PORT"
echo "Drop your results JSON files onto the page to analyze them."
echo "Ctrl-C to stop."
echo

if [ -d "$RESULTS_DIR" ]; then
    if command -v open >/dev/null 2>&1; then
        open "$RESULTS_DIR"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$RESULTS_DIR" >/dev/null 2>&1 &
    fi
fi

exec npm --prefix "$DASHBOARD_DIR" run preview -- --port "$PORT" --open
