#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/qualification_python.sh"
PATH="${HOME}/.local/bin:${PATH}"
if [ -d /usr/lib/wsl/lib ]; then PATH="/usr/lib/wsl/lib:${PATH}"; fi
QUALIFICATION_PYTHON="$(qualification_python || true)"
VENV="$ROOT/qualification-env"

usage() {
    echo "Usage: $0 TARGET BASELINE_VERSION TARGET_VERSION [OUTPUT_DIR] [--execute]"
    echo "       $0 [--execute] [--vllm-only]"
    echo "       $0 --list-targets"
}

if [ "${1:-}" = "--list-targets" ]; then
    if [ -x "$VENV/bin/python" ]; then
        exec "$VENV/bin/python" -m scripts.release.qualification_recipe --list-targets
    fi
    if [ -z "$QUALIFICATION_PYTHON" ]; then
        echo "Python 3.11 or newer is required; run bootstrap_qualification.sh --execute." >&2
        exit 1
    fi
    exec "$QUALIFICATION_PYTHON" -m scripts.release.qualification_recipe --list-targets
fi
if [ "$#" -eq 0 ] || [ "${1:-}" = "--execute" ] || [ "${1:-}" = "--vllm-only" ]; then
    EXECUTE_AUTO=false
    VLLM_ONLY=false
    for ARG in "$@"; do
        case "$ARG" in
            --execute) EXECUTE_AUTO=true ;;
            --vllm-only) VLLM_ONLY=true ;;
            *) usage; exit 2 ;;
        esac
    done
    if [ "$EXECUTE_AUTO" = true ]; then
        "$ROOT/bootstrap_qualification.sh" --execute
    else
        "$ROOT/bootstrap_qualification.sh"
    fi
    QUALIFICATION_PYTHON="$(qualification_python || true)"
    if [ -z "$QUALIFICATION_PYTHON" ]; then
        echo "Bootstrap did not provide Python 3.11 or newer." >&2
        exit 1
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        "$QUALIFICATION_PYTHON" -m venv "$VENV"
        "$VENV/bin/python" -m pip install --upgrade pip
    fi
    "$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements.txt"
    cd "$ROOT"
    COMMAND=("$VENV/bin/python" -m scripts.release.qualification_auto --root "$ROOT")
    if [ "$EXECUTE_AUTO" = true ]; then COMMAND+=(--execute); fi
    if [ "$VLLM_ONLY" = true ]; then COMMAND+=(--vllm-only); fi
    exec "${COMMAND[@]}"
fi
if [ "$#" -lt 3 ]; then
    usage
    exit 2
fi

if [ -z "$QUALIFICATION_PYTHON" ]; then
    echo "Python 3.11 or newer is required; run bootstrap_qualification.sh --execute." >&2
    exit 1
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
fi
"$VENV/bin/python" -m pip install --quiet -r "$ROOT/requirements.txt"

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
