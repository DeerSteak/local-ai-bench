#!/usr/bin/env bash
set -u

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_ROOT" || exit 1

bash run_bench.sh --ui gui
status=$?

if [ "$status" -eq 0 ]; then
    exit 0
fi

echo
echo "Local AI Bench stopped with an error. Review the messages above."
read -r -p "Press Enter to close this window."

exit "$status"
