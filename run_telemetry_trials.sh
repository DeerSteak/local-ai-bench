#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/bench-env/bin/python"
MODEL=""
ENGINE="llamacpp"
TELEMETRY_MODE="memory"
WORKLOAD="llm"
SUSTAINED_DURATION=120
AMBIENT_TEMP=""
PAIRS=20
MIN_QUALIFICATION_PAIRS=20
INTERVAL="0.5"
WAIT_SECONDS=30
OUT_DIR=""
DRY_RUN=false

usage() {
    echo "Usage: bash run_telemetry_trials.sh --model TAG [--engine NAME] [--pairs N]"
    echo "       [--telemetry memory|power|temperature] [--workload llm|sustained]"
    echo "       [--sustained-duration SEC --ambient-temp-c C] [--interval SEC] [--wait SEC]"
    echo "       [--out-dir DIR] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) MODEL="${2:-}"; shift 2 ;;
        --engine) ENGINE="${2:-}"; shift 2 ;;
        --telemetry) TELEMETRY_MODE="${2:-}"; shift 2 ;;
        --workload) WORKLOAD="${2:-}"; shift 2 ;;
        --sustained-duration) SUSTAINED_DURATION="${2:-}"; shift 2 ;;
        --ambient-temp-c) AMBIENT_TEMP="${2:-}"; shift 2 ;;
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
if [ "$WORKLOAD" != "llm" ] && [ "$WORKLOAD" != "sustained" ]; then
    echo "--workload must be llm or sustained" >&2
    exit 2
fi
if [ "$WORKLOAD" = "sustained" ] && [ "$TELEMETRY_MODE" != "temperature" ]; then
    echo "--workload sustained requires --telemetry temperature" >&2
    exit 2
fi
if [ "$WORKLOAD" = "sustained" ] && { ! [[ "$SUSTAINED_DURATION" =~ ^[1-9][0-9]*$ ]] || (( SUSTAINED_DURATION < 120 )); }; then
    echo "--sustained-duration must be an integer of at least 120 seconds" >&2
    exit 2
fi
if [ "$WORKLOAD" = "sustained" ] && [ -z "$AMBIENT_TEMP" ]; then
    echo "--ambient-temp-c is required for sustained qualification" >&2
    exit 2
fi
if [ "$TELEMETRY_MODE" != "memory" ] && [ "$TELEMETRY_MODE" != "power" ] && [ "$TELEMETRY_MODE" != "temperature" ]; then
    echo "--telemetry must be memory, power, or temperature" >&2
    exit 2
fi
if [ "$TELEMETRY_MODE" = "temperature" ] && [ "$(uname -s)" != "Linux" ] && [ "$DRY_RUN" = false ]; then
    echo "Temperature qualification currently requires Linux sensor sources." >&2
    exit 1
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
    local temperature_flag=0
    if [ "$mode" = "on" ]; then temperature_flag=1; fi
    local command=(bash "$SCRIPT_DIR/run_bench.sh" --ui none --engine "$ENGINE"
        --tests "$WORKLOAD" --llm-models "$MODEL" --warmup 2 --out "$output")
    if [ "$WORKLOAD" = "llm" ]; then
        command+=(--max-prompt-tokens 2048 --runs 3)
    else
        command+=(--sustained-duration "$SUSTAINED_DURATION" --ambient-temp-c "$AMBIENT_TEMP")
    fi
    if [ "$TELEMETRY_MODE" = "temperature" ]; then
        command+=(--memory-telemetry --power-telemetry)
    elif [ "$TELEMETRY_MODE" = "power" ]; then
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
        if [ "$TELEMETRY_MODE" = "temperature" ]; then
            printf 'LOCAL_AI_BENCH_QUALIFICATION_TEMPERATURE=%q ' "$temperature_flag"
        fi
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
    if [ "$TELEMETRY_MODE" = "temperature" ]; then
        LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC="$INTERVAL" \
            LOCAL_AI_BENCH_QUALIFICATION_TEMPERATURE="$temperature_flag" \
            "${command[@]}"
    else
        LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC="$INTERVAL" "${command[@]}"
    fi
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

"$PYTHON" - "$OUT_DIR" "$PAIRS" "$INTERVAL" "$TELEMETRY_MODE" "$WORKLOAD" <<'PY'
import json
import platform
import sys
from pathlib import Path

directory, pair_count, interval = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
telemetry_mode = sys.argv[4]
workload = sys.argv[5]
first = json.loads((directory / "off-01.json").read_text(encoding="utf-8"))
first_on = json.loads((directory / "on-01.json").read_text(encoding="utf-8"))
section = "sustained" if workload == "sustained" else "llm"
models = list((first.get(section) or {}).keys())
if len(models) != 1:
    raise SystemExit(f"Expected exactly one {section} result model when building the manifest")
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
    "source": (
        first_on.get("run", {}).get("effective_config", {}).get("temperature_sources")
        if telemetry_mode == "temperature" else
        first_on.get("run", {}).get("effective_config", {}).get(f"{telemetry_mode}_source")
    ),
    "scope": "sensor_channels" if telemetry_mode == "temperature" else
        first_on.get("run", {}).get("effective_config", {}).get(f"{telemetry_mode}_scope"),
    "section": section,
    "model": models[0],
    "case": "soak" if workload == "sustained" else "2K",
    "pairs": pairs,
}
(directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

if (( PAIRS < MIN_QUALIFICATION_PAIRS )); then
    echo "Smoke evidence: $OUT_DIR (qualification requires $MIN_QUALIFICATION_PAIRS pairs)"
    exit 0
fi

set +e
"$PYTHON" -m scripts.release.telemetry_qualification \
    "$OUT_DIR/manifest.json" --output "$OUT_DIR/report.json"
analysis_status=$?
set -e
if (( analysis_status != 0 )); then
    echo "Qualification screen rejected: $OUT_DIR" >&2
    exit 3
fi
echo "Qualification evidence passed: $OUT_DIR"
