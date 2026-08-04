#!/usr/bin/env bash
set -u

SCRIPT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_ROOT" || exit 1

bash setup.sh --interface gui
status=$?

if [ "$status" -eq 10 ]; then
    exit 0
fi

if [ "$status" -ne 0 ]; then
    echo
    echo "Setup stopped with an error. Review the messages above."
    read -r -p "Press Enter to close this window."
fi

exit "$status"
