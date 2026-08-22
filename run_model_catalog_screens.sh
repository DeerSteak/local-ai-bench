#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATES=(
    "qwen3.8-27b"
    "muse-glimmer-30b"
    "nemotron-3.5-lightning-30b-a3b"
    "gemma-4-26b-a4b"
)
ENGINES=("llamacpp" "vllm")
ENGINE_FILTER=""
OUTPUT_ROOT=""
LIST_ONLY=0

usage() {
    echo "Usage: $0 [--engine llamacpp|vllm] [--output-root DIR] [--list]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --engine)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            ENGINE_FILTER="$2"
            if [ "$ENGINE_FILTER" != "llamacpp" ] && [ "$ENGINE_FILTER" != "vllm" ]; then
                usage
                exit 2
            fi
            shift 2
            ;;
        --output-root)
            [ "$#" -ge 2 ] || { usage; exit 2; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

selected_engines() {
    if [ -n "$ENGINE_FILTER" ]; then
        echo "$ENGINE_FILTER"
    else
        printf '%s\n' "${ENGINES[@]}"
    fi
}

if [ "$LIST_ONLY" -eq 1 ]; then
    for candidate in "${CANDIDATES[@]}"; do
        while IFS= read -r engine; do
            printf '%s\t%s\n' "$candidate" "$engine"
        done < <(selected_engines)
    done
    exit 0
fi

PYTHON="$ROOT/bench-env/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "bench-env is missing; run setup.sh before catalog screening." >&2
    exit 2
fi

trap 'echo; echo "Catalog screen matrix interrupted." >&2; exit 130' INT TERM

passed=()
failed=()
for candidate in "${CANDIDATES[@]}"; do
    while IFS= read -r engine; do
        echo
        echo "=== Catalog screen: $candidate / $engine ==="
        command=(
            "$PYTHON" -m scripts.release.model_catalog_screen
            --candidate "$candidate" --engine "$engine" --execute
        )
        if [ -n "$OUTPUT_ROOT" ]; then
            command+=(--output-root "$OUTPUT_ROOT")
        fi
        if PYTHONUNBUFFERED=1 "${command[@]}"; then
            passed+=("$candidate/$engine")
        else
            exit_code=$?
            failed+=("$candidate/$engine (exit $exit_code)")
        fi
    done < <(selected_engines)
done

echo
echo "Catalog screen matrix complete: ${#passed[@]} passed, ${#failed[@]} failed."
if [ "${#failed[@]}" -gt 0 ]; then
    printf '  FAILED: %s\n' "${failed[@]}" >&2
    exit 1
fi
