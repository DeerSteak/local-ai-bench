#!/usr/bin/env bash
set -u

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_ROOT" || exit 1
LAUNCHER_TTY="$(tty 2>/dev/null || true)"

bash run_bench.sh --ui gui
status=$?

if [ "$status" -eq 0 ]; then
    if [ -n "$LAUNCHER_TTY" ]; then
        nohup /usr/bin/osascript "$SCRIPT_ROOT/scripts/close_terminal_tab.applescript" \
            "$LAUNCHER_TTY" >/dev/null 2>&1 &
    fi
    exit 0
fi

echo
echo "Local AI Bench stopped with an error. Review the messages above."
read -r -p "Press Enter to close this window."

exit "$status"
