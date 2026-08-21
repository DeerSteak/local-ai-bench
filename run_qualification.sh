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
    bash "$ROOT/setup.sh" --qualification "$ENGINE" --qualification-target "$TARGET"
} 2>&1 | tee -a "$SETUP_LOG"
SETUP_EXIT=${PIPESTATUS[0]}
set -e
if [ "$SETUP_EXIT" -eq 0 ]; then SETUP_RESULT="passed"; else SETUP_RESULT="failed"; fi
FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{\n  "schema": "qualification-setup-v1",\n  "target": "%s",\n  "status": "%s",\n  "exit_code": %d,\n  "started_at": "%s",\n  "finished_at": "%s",\n  "log": "setup.log"\n}\n' \
    "$TARGET" "$SETUP_RESULT" "$SETUP_EXIT" "$STARTED_AT" "$FINISHED_AT" \
    > "$SETUP_STATUS"
chmod 0644 "$SETUP_LOG" "$SETUP_STATUS"
if [ "$SETUP_EXIT" -ne 0 ]; then
    echo "Qualification setup failed; evidence saved to $SETUP_LOG" >&2
    exit "$SETUP_EXIT"
fi
exec "$ROOT/bench-env/bin/python" -m scripts.release.qualification_run \
    "$TARGET" --root "$ROOT" --result "$RESULT"
