#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${PYTHON:-}" ]; then
    QUALIFICATION_PYTHON="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
    QUALIFICATION_PYTHON="python3.12"
else
    QUALIFICATION_PYTHON="python3"
fi
VENV="$ROOT/bench-env"

usage() {
    echo "Usage: $0 TARGET BASELINE_VERSION TARGET_VERSION [OUTPUT_DIR] [--execute]"
    echo "       $0 --list-targets"
}

if [ "${1:-}" = "--list-targets" ]; then
    if [ -x "$VENV/bin/python" ]; then
        exec "$VENV/bin/python" -m scripts.release.qualification_recipe --list-targets
    fi
    exec "$QUALIFICATION_PYTHON" -m scripts.release.qualification_recipe --list-targets
fi
if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "--execute" ]; }; then
    if [ "${1:-}" = "--execute" ]; then
        "$ROOT/bootstrap_qualification.sh" --execute
    else
        "$ROOT/bootstrap_qualification.sh"
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        "$QUALIFICATION_PYTHON" -m venv "$VENV"
        "$VENV/bin/python" -m pip install --upgrade pip
        "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
    fi
    cd "$ROOT"
    COMMAND=("$VENV/bin/python" -m scripts.release.qualification_auto --root "$ROOT")
    if [ "${1:-}" = "--execute" ]; then COMMAND+=(--execute); fi
    exec "${COMMAND[@]}"
fi
if [ "$#" -lt 3 ]; then
    usage
    exit 2
fi

TARGET="$1"
BASELINE_VERSION="$2"
TARGET_VERSION="$3"
OUTPUT_DIR="${4:-$ROOT/qualification-evidence/$TARGET}"
EXECUTE="${5:-}"
if [ "$EXECUTE" != "" ] && [ "$EXECUTE" != "--execute" ]; then
    usage
    exit 2
fi

if [ ! -x "$VENV/bin/python" ]; then
    "$QUALIFICATION_PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi

cd "$ROOT"
"$VENV/bin/python" -m scripts.release.qualification_recipe \
    --target "$TARGET" --root "$ROOT" --output "$OUTPUT_DIR" \
    --baseline-version "$BASELINE_VERSION" --target-version "$TARGET_VERSION"

COMMAND=("$VENV/bin/python" -m scripts.release.qualification_automation
    "$OUTPUT_DIR/qualification-recipe.json" --output "$OUTPUT_DIR")
if [ "$EXECUTE" = "--execute" ]; then
    COMMAND+=(--execute)
fi
exec "${COMMAND[@]}"
