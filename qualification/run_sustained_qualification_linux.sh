#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL=""
AMBIENT=""
DURATION=600
REPEATS=1
WAIT_SECONDS=300
OUT_DIR=""
DRY_RUN=false

usage() {
    echo "Usage: bash qualification/run_sustained_qualification_linux.sh --model TAG --ambient-temp-c C"
    echo "       [--duration SEC] [--repeats N] [--wait SEC] [--out-dir DIR] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --ambient-temp-c) AMBIENT="${2:-}"; shift 2 ;;
        --duration) DURATION="${2:-}"; shift 2 ;;
        --repeats) REPEATS="${2:-}"; shift 2 ;;
        --wait) WAIT_SECONDS="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$AMBIENT" ]; then
    echo "--model and --ambient-temp-c are required" >&2
    exit 2
fi
if ! [[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || (( DURATION < 120 )); then
    echo "--duration must be an integer of at least 120 seconds" >&2
    exit 2
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repeats must be a positive integer" >&2
    exit 2
fi
if ! [[ "$WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "--wait must be a non-negative integer" >&2
    exit 2
fi
if [ "$(uname -s)" != "Linux" ] && [ "$DRY_RUN" = false ]; then
    echo "This qualification wrapper is for Linux systems." >&2
    exit 1
fi
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$ROOT/results/qualification/sustained-linux-$(date '+%Y%m%d-%H%M%S')"
fi
if [ "$DRY_RUN" = false ] && [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
    echo "Qualification requires a clean Git worktree; run git status --short." >&2
    exit 1
fi

for ((run=1; run<=REPEATS; run++)); do
    label="$(printf '%02d' "$run")"
    command=(bash "$ROOT/run_bench.sh" --ui none --tests sustained
        --llm-models "$MODEL" --memory-telemetry --power-telemetry
        --sustained-duration "$DURATION" --ambient-temp-c "$AMBIENT"
        --out "$OUT_DIR/run-$label.json")
    if [ "$DRY_RUN" = true ]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        mkdir -p "$OUT_DIR"
        "${command[@]}"
        if (( run < REPEATS )); then
            sleep "$WAIT_SECONDS"
        fi
    fi
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry run only; no benchmark was launched."
else
    echo "Sustained qualification evidence: $OUT_DIR"
fi
