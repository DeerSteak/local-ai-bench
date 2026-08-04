#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/bench-env"
cd "$SCRIPT_DIR"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "[$(date '+%H:%M:%S')] Virtual environment not found at $VENV — run setup.sh first."
    exit 1
fi

source "$VENV/bin/activate"
if [ "$#" -eq 0 ]; then
    exec python -m scripts.app.benchmark_launcher --ui auto
fi
if [ "$1" = "--ui" ] || [ "$1" = "--interface" ]; then
    exec python -m scripts.app.benchmark_launcher "$@"
fi
exec python -m scripts.app.benchmark "$@"
