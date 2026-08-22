#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--list-targets" ]; then
    if [ -x "$ROOT/bench-env/bin/python" ]; then
        exec "$ROOT/bench-env/bin/python" -m scripts.release.qualification_run --list-targets
    fi
    sed -n 's/^    {"id": "\([^"]*\)".*$/\1/p' "$ROOT/scripts/release/qualification_targets.py"
    exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 TARGET [RESULT_JSON]" >&2
    echo "       $0 --list-targets" >&2
    exit 2
fi

TARGET="$1"
RESULT="${2:-$ROOT/qualification-evidence/$TARGET/results_qualification_${TARGET}.json}"
if ! grep -Fq "{\"id\": \"$TARGET\"," "$ROOT/scripts/release/qualification_targets.py"; then
    echo "Unknown qualification target: $TARGET" >&2
    exit 2
fi
ENGINE="llamacpp"
if [[ "$TARGET" == *-vllm-* ]]; then ENGINE="vllm"; fi

EVIDENCE_DIR="$(dirname "$RESULT")"
SETUP_LOG="$EVIDENCE_DIR/setup.log"
SETUP_STATUS="$EVIDENCE_DIR/setup-status.json"
mkdir -p "$EVIDENCE_DIR"
chmod 0755 "$EVIDENCE_DIR"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{\n  "schema": "qualification-setup-v1",\n  "target": "%s",\n  "status": "running",\n  "exit_code": null,\n  "started_at": "%s",\n  "finished_at": null,\n  "log": "setup.log"\n}\n' \
    "$TARGET" "$STARTED_AT" > "$SETUP_STATUS"
set +e
{
    echo
    echo "=== qualification setup attempt $STARTED_AT ==="
} >> "$SETUP_LOG"
LOG_OFFSET=$(( $(wc -c < "$SETUP_LOG") + 1 ))
LOG_FOLLOWER_PID=""
stop_log_follower() {
    if [ -n "$LOG_FOLLOWER_PID" ]; then
        kill "$LOG_FOLLOWER_PID" 2>/dev/null || true
        wait "$LOG_FOLLOWER_PID" 2>/dev/null || true
        LOG_FOLLOWER_PID=""
    fi
}
trap stop_log_follower EXIT INT TERM
tail -c +"$LOG_OFFSET" -f "$SETUP_LOG" &
LOG_FOLLOWER_PID=$!
PYTHONUNBUFFERED=1 bash "$ROOT/setup.sh" \
    --qualification "$ENGINE" --qualification-target "$TARGET" >> "$SETUP_LOG" 2>&1
SETUP_EXIT=$?
sleep 1
stop_log_follower
set -e
if [ "$SETUP_EXIT" -eq 0 ]; then
    SETUP_RESULT="passed"
elif [ "$SETUP_EXIT" -eq 75 ]; then
    SETUP_RESULT="reboot_required"
else
    SETUP_RESULT="failed"
fi
FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{\n  "schema": "qualification-setup-v1",\n  "target": "%s",\n  "status": "%s",\n  "exit_code": %d,\n  "started_at": "%s",\n  "finished_at": "%s",\n  "log": "setup.log"\n}\n' \
    "$TARGET" "$SETUP_RESULT" "$SETUP_EXIT" "$STARTED_AT" "$FINISHED_AT" \
    > "$SETUP_STATUS"
chmod 0644 "$SETUP_LOG" "$SETUP_STATUS"
if [ "$SETUP_EXIT" -ne 0 ]; then
    if [ "$SETUP_EXIT" -eq 75 ]; then
        echo "NVIDIA driver installation completed. Reboot, then rerun this command." >&2
        exit "$SETUP_EXIT"
    fi
    echo "Qualification setup failed; evidence saved to $SETUP_LOG" >&2
    exit "$SETUP_EXIT"
fi
exec "$ROOT/bench-env/bin/python" -m scripts.release.qualification_run \
    "$TARGET" --root "$ROOT" --result "$RESULT"
