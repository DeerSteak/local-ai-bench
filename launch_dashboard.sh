#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$SCRIPT_DIR/dashboard"
RESULTS_DIR="$SCRIPT_DIR/results"
PORT=3000
SELECTED_RESULTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --port"
                exit 1
            fi
            PORT="$2"
            shift 2
            ;;
        --result)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --result"
                exit 1
            fi
            SELECTED_RESULTS+=("$2")
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

OPEN_PATH="/"
if [ ${#SELECTED_RESULTS[@]} -gt 0 ]; then
    OPEN_PATH="/?autoload=1"
fi

# Stage first so an already-running workspace server reopens with this selection.
node "$DASHBOARD_DIR/stage_selected_results.mjs" "$DASHBOARD_DIR/dist" "${SELECTED_RESULTS[@]+"${SELECTED_RESULTS[@]}"}"
if (cd "$SCRIPT_DIR" && "$SCRIPT_DIR/bench-env/bin/python" -m scripts.app.dashboard_reuse \
        --port "$PORT" --open-path "$OPEN_PATH"); then
    echo "Dashboard already running -> http://localhost:$PORT$OPEN_PATH"
    exit 0
fi

echo "Building dashboard ..."
(cd "$DASHBOARD_DIR" && npm run build)
# "${arr[@]}" on a genuinely empty array raises "unbound variable" under `set -u` on
# bash < 4.4 (e.g. macOS's stock bash 3.2) — this expansion works on both old and new bash.
node "$DASHBOARD_DIR/stage_selected_results.mjs" "$DASHBOARD_DIR/dist" "${SELECTED_RESULTS[@]+"${SELECTED_RESULTS[@]}"}"
echo "Build complete."
echo

echo "Dashboard -> http://localhost:$PORT"
echo "Drop your results JSON files onto the page to analyze them."
echo "Ctrl-C to stop."
echo

if [ ${#SELECTED_RESULTS[@]} -eq 0 ] && [ -d "$RESULTS_DIR" ]; then
    if command -v open >/dev/null 2>&1; then
        open "$RESULTS_DIR"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$RESULTS_DIR" >/dev/null 2>&1 &
    fi
fi

cleanup_selected_results() {
    node "$DASHBOARD_DIR/stage_selected_results.mjs" "$DASHBOARD_DIR/dist" >/dev/null 2>&1 || true
}
trap cleanup_selected_results EXIT

(cd "$SCRIPT_DIR" && "$SCRIPT_DIR/bench-env/bin/python" -m scripts.app.workspace_server \
    --dist "$DASHBOARD_DIR/dist" --port "$PORT" --open-path "$OPEN_PATH")
