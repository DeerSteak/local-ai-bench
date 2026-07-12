#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/bench-env"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Virtual environment not found at $VENV — run setup.sh first."
    exit 1
fi

source "$VENV/bin/activate"
exec python "$SCRIPT_DIR/scripts/benchmark.py" "$@"
