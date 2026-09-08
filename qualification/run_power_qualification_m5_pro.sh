#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="gemma3:1b-it-q4_K_M"
OUT_ROOT=""
DRY_RUN=false

usage() {
    echo "Usage: bash qualification/run_power_qualification_m5_pro.sh [--out-root DIR] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ "$(uname -s)" != "Darwin" ] && [ "$DRY_RUN" = false ]; then
    echo "This qualification wrapper is for the macOS M5 Pro evidence run." >&2
    exit 1
fi
if [ -z "$OUT_ROOT" ]; then
    OUT_ROOT="$ROOT/results/qualification/power-m5-pro-pre4-$(date '+%Y%m%d-%H%M%S')"
fi

for interval in 0.25 0.5 1.0; do
    command=(bash "$SCRIPT_DIR/run_telemetry_trials.sh"
        --model "$MODEL" --engine llamacpp --telemetry power
        --pairs 20 --interval "$interval" --wait 30
        --out-dir "$OUT_ROOT/$interval")
    if [ "$DRY_RUN" = true ]; then
        command+=(--dry-run)
    elif command -v caffeinate >/dev/null 2>&1; then
        command=(caffeinate -dimsu "${command[@]}")
    fi
    "${command[@]}"
done

echo "M5 Pro power qualification evidence: $OUT_ROOT"
