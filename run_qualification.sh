#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--list-targets" ]; then
    if [ -x "$ROOT/bench-env/bin/python" ]; then
        exec "$ROOT/bench-env/bin/python" -m scripts.release.qualification_run --list-targets
    fi
    sed -n 's/^    "\([^"]*\)": ".*",$/\1/p' "$ROOT/scripts/release/qualification_targets.py"
    exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 TARGET [RESULT_JSON]" >&2
    echo "       $0 --list-targets" >&2
    exit 2
fi

TARGET="$1"
RESULT="${2:-$ROOT/results_qualification_${TARGET}.json}"
if ! grep -Fq "\"$TARGET\":" "$ROOT/scripts/release/qualification_targets.py"; then
    echo "Unknown qualification target: $TARGET" >&2
    exit 2
fi
ENGINE="llamacpp"
if [[ "$TARGET" == *-vllm-* ]]; then ENGINE="vllm"; fi

bash "$ROOT/setup.sh" --qualification "$ENGINE"
exec "$ROOT/bench-env/bin/python" -m scripts.release.qualification_run \
    "$TARGET" --root "$ROOT" --result "$RESULT"
