#!/usr/bin/env bash
set -u

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_ROOT" || exit 1

bash launch_dashboard.sh
status=$?

if [ "$status" -ne 0 ] && [ "$status" -ne 130 ]; then
    echo
    echo "The Local AI Bench dashboard stopped with an error. Review the messages above."
    read -r -p "Press Enter to close this window."
fi

exit "$status"
