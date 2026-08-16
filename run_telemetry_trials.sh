#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/bench-env/bin/python"
MODEL=""
ENGINE="llamacpp"
TELEMETRY_MODE="memory"
PAIRS=20
MIN_QUALIFICATION_PAIRS=20
INTERVAL="0.5"
WAIT_SECONDS=30
OUT_DIR=""
DRY_RUN=false

usage() {
    echo "Usage: bash run_telemetry_trials.sh --model TAG [--engine NAME] [--pairs N]"
    echo "       [--telemetry memory|power] [--interval SEC] [--wait SEC] [--out-dir DIR] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --engine) ENGINE="${2:-}"; shift 2 ;;
        --telemetry) TELEMETRY_MODE="${2:-}"; shift 2 ;;
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
if [ "$TELEMETRY_MODE" != "memory" ] && [ "$TELEMETRY_MODE" != "power" ]; then
    echo "--telemetry must be memory or power" >&2
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
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$SCRIPT_DIR/results/qualification/$TELEMETRY_MODE-$(date '+%Y%m%d-%H%M%S')"
fi
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

SUDO_KEEPALIVE_PID=""
cleanup_sudo_keepalive() {
    if [ -n "$SUDO_KEEPALIVE_PID" ]; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    fi
}
trap cleanup_sudo_keepalive EXIT

if [ "$TELEMETRY_MODE" = "power" ] && [ "$DRY_RUN" = false ] && [ "$(uname -s)" = "Darwin" ]; then
    echo "Power qualification needs temporary administrator permission for powermetrics."
    sudo -v
    (while sudo -n -v 2>/dev/null; do sleep 60; done) &
    SUDO_KEEPALIVE_PID=$!
fi

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
    if [ "$TELEMETRY_MODE" = "power" ]; then
        command+=(--memory-telemetry)
        if [ "$mode" = "on" ]; then
            command+=(--power-telemetry)
        fi
    elif [ "$mode" = "on" ]; then
        command+=(--memory-telemetry)
    else
        command+=(--no-memory-telemetry)
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

"$PYTHON" - "$OUT_DIR" "$PAIRS" "$INTERVAL" "$TELEMETRY_MODE" <<'PY'
import json
import platform
import sys
from pathlib import Path

directory, pair_count, interval = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
telemetry_mode = sys.argv[4]
first = json.loads((directory / "off-01.json").read_text(encoding="utf-8"))
first_on = json.loads((directory / "on-01.json").read_text(encoding="utf-8"))
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
    "telemetry_mode": telemetry_mode,
    "source": first_on.get("run", {}).get("effective_config", {}).get(
        f"{telemetry_mode}_source"
    ),
    "scope": first_on.get("run", {}).get("effective_config", {}).get(
        f"{telemetry_mode}_scope"
    ),
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
