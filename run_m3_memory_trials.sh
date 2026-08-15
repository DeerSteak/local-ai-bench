#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/bench-env/bin/python"
MODEL=""
ENGINE="llamacpp"
PAIRS=20
MIN_QUALIFICATION_PAIRS=20
INTERVAL="0.5"
WAIT_SECONDS=30
OUT_DIR="$SCRIPT_DIR/results/qualification/m3-memory-$(date '+%Y%m%d-%H%M%S')"
DRY_RUN=false

usage() {
    echo "Usage: bash run_m3_memory_trials.sh --model TAG [--engine NAME] [--pairs N]"
    echo "       [--interval SEC] [--wait SEC] [--out-dir DIR] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --engine) ENGINE="${2:-}"; shift 2 ;;
        --pairs) PAIRS="${2:-}"; shift 2 ;;
        --interval) INTERVAL="${2:-}"; shift 2 ;;
        --wait) WAIT_SECONDS="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "--model is required" >&2
    exit 2
fi
if ! [[ "$PAIRS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "--pairs must be positive and --wait must be non-negative" >&2
    exit 2
fi
if [ ! -x "$PYTHON" ]; then
    echo "Virtual environment not found at $PYTHON — run setup.sh first." >&2
    exit 1
fi
if [ "$DRY_RUN" = false ] && [ -n "$(git -C "$SCRIPT_DIR" status --porcelain)" ]; then
    echo "Qualification requires a clean Git worktree; run git status --short." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

is_complete() {
    "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if data.get("run", {}).get("status") == "complete" else 1)
PY
}

run_trial() {
    local mode="$1" pair="$2" output="$3"
    local command=(bash "$SCRIPT_DIR/run_bench.sh" --ui none --engine "$ENGINE" --tests llm
        --llm-models "$MODEL" --max-prompt-tokens 2048 --warmup 2 --runs 3
        --out "$output")
    if [ "$mode" = "on" ]; then
        command+=(--memory-telemetry)
    fi
    if [ "$DRY_RUN" = true ]; then
        printf 'LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC=%q ' "$INTERVAL"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    if [ -e "$output" ]; then
        if is_complete "$output"; then
            echo "[$(date '+%H:%M:%S')] Pair $pair telemetry-$mode already complete — skipping"
            return
        fi
        echo "Incomplete output exists at $output; move it aside before retrying." >&2
        exit 1
    fi
    echo "[$(date '+%H:%M:%S')] Pair $pair telemetry-$mode"
    LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC="$INTERVAL" "${command[@]}"
}

invocation=0
for ((pair=1; pair<=PAIRS; pair++)); do
    pair_label="$(printf '%02d' "$pair")"
    modes=(off on)
    if (( pair % 2 == 0 )); then modes=(on off); fi
    for mode in "${modes[@]}"; do
        run_trial "$mode" "$pair_label" "$OUT_DIR/$mode-$pair_label.json"
        invocation=$((invocation + 1))
        if [ "$DRY_RUN" = false ] && (( invocation < PAIRS * 2 )); then
            sleep "$WAIT_SECONDS"
        fi
    done
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry run only; no benchmark or manifest was written."
    exit 0
fi

"$PYTHON" - "$OUT_DIR" "$PAIRS" "$INTERVAL" <<'PY'
import json
import platform
import sys
from pathlib import Path

directory, pair_count, interval = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
first = json.loads((directory / "off-01.json").read_text(encoding="utf-8"))
models = list((first.get("llm") or {}).keys())
if len(models) != 1:
    raise SystemExit("Expected exactly one LLM result model when building the manifest")
pairs = []
for index in range(1, pair_count + 1):
    label = f"{index:02d}"
    pairs.append({
        "order": "off-on" if index % 2 else "on-off",
        "off": f"off-{label}.json",
        "on": f"on-{label}.json",
    })
manifest = {
    "platform": platform.platform(),
    "interval_sec": interval,
    "section": "llm",
    "model": models[0],
    "case": "2K",
    "pairs": pairs,
}
(directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

if (( PAIRS < MIN_QUALIFICATION_PAIRS )); then
    echo "Smoke evidence: $OUT_DIR (qualification requires $MIN_QUALIFICATION_PAIRS pairs)"
    exit 0
fi

"$PYTHON" -m scripts.release.telemetry_qualification \
    "$OUT_DIR/manifest.json" --output "$OUT_DIR/report.json"
echo "Qualification evidence: $OUT_DIR"
