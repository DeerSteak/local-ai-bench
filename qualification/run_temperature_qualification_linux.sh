#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL=""
AMBIENT_TEMP=""
PAIRS=20
SUSTAINED_DURATION=120
LATENCY_WAIT=30
SUSTAINED_WAIT=120
SUITE_WAIT=300
OUT_ROOT=""
DRY_RUN=false
INTERVALS=(0.25 0.5 1.0)

usage() {
    echo "Usage: bash qualification/run_temperature_qualification_linux.sh --model TAG --ambient-temp-c C"
    echo "       [--pairs N] [--sustained-duration SEC] [--out-root DIR] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --ambient-temp-c) AMBIENT_TEMP="${2:-}"; shift 2 ;;
        --pairs) PAIRS="${2:-}"; shift 2 ;;
        --sustained-duration) SUSTAINED_DURATION="${2:-}"; shift 2 ;;
        --out-root) OUT_ROOT="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$AMBIENT_TEMP" ]; then
    echo "--model and --ambient-temp-c are required" >&2
    exit 2
fi
if ! [[ "$PAIRS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--pairs must be a positive integer" >&2
    exit 2
fi
if ! [[ "$SUSTAINED_DURATION" =~ ^[1-9][0-9]*$ ]] || (( SUSTAINED_DURATION < 120 )); then
    echo "--sustained-duration must be an integer of at least 120 seconds" >&2
    exit 2
fi
if [ "$(uname -s)" != "Linux" ] && [ "$DRY_RUN" = false ]; then
    echo "Temperature qualification currently requires Linux sensor sources." >&2
    exit 1
fi
if [ -z "$OUT_ROOT" ]; then
    OUT_ROOT="$ROOT/results/qualification/temperature-linux-$(date '+%Y%m%d-%H%M%S')"
fi

qualification_failed=false
run_screen() {
    local workload="$1" interval="$2" wait_seconds="$3" output="$4"
    local command=(bash "$SCRIPT_DIR/run_telemetry_trials.sh"
        --model "$MODEL" --telemetry temperature --workload "$workload"
        --pairs "$PAIRS" --interval "$interval" --wait "$wait_seconds"
        --out-dir "$output")
    if [ "$workload" = "sustained" ]; then
        command+=(--sustained-duration "$SUSTAINED_DURATION" --ambient-temp-c "$AMBIENT_TEMP")
    fi
    if [ "$DRY_RUN" = true ]; then command+=(--dry-run); fi
    set +e
    "${command[@]}"
    local status=$?
    set -e
    if (( status == 3 )); then
        qualification_failed=true
    elif (( status != 0 )); then
        exit "$status"
    fi
}

for interval in "${INTERVALS[@]}"; do
    run_screen llm "$interval" "$LATENCY_WAIT" "$OUT_ROOT/latency/$interval"
done

if [ "$DRY_RUN" = false ]; then sleep "$SUITE_WAIT"; fi

for index in "${!INTERVALS[@]}"; do
    interval="${INTERVALS[$index]}"
    run_screen sustained "$interval" "$SUSTAINED_WAIT" "$OUT_ROOT/sustained/$interval"
    if [ "$DRY_RUN" = false ] && (( index + 1 < ${#INTERVALS[@]} )); then
        sleep "$SUITE_WAIT"
    fi
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry run only; previewed all latency and sustained interval screens."
elif [ "$qualification_failed" = true ]; then
    echo "All screens completed; at least one observer-effect report rejected. Evidence: $OUT_ROOT" >&2
    exit 3
else
    echo "All temperature qualification screens passed: $OUT_ROOT"
fi
