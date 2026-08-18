#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SYSTEM="$(uname -s)"
if [ "$SYSTEM" = "Darwin" ]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew is required to install Python 3.12 unattended; install it from https://brew.sh and rerun." >&2
        exit 1
    fi
    echo "Qualification host bootstrap preview:"
    echo "  brew install python@3.12 git"
    if [ "$MODE" != "--execute" ]; then
        echo "Repeat with --execute after reviewing the command."
        exit 0
    fi
    brew install python@3.12 git
    python3.12 --version
    git --version
    exit 0
fi

if command -v apt-get >/dev/null 2>&1; then
    COMMAND=(sudo -n apt-get install -y git cmake build-essential python3 python3-venv python3.12 python3.12-venv)
    UPDATE=(sudo -n apt-get update)
elif command -v dnf >/dev/null 2>&1; then
    COMMAND=(sudo -n dnf install -y git cmake gcc-c++ make python3 python3.12)
    UPDATE=()
else
    echo "Unsupported host package manager; install Git, CMake, a C++ compiler, Python 3, and Python 3.12 for vLLM targets." >&2
    exit 1
fi

echo "Qualification host bootstrap preview:"
if [ "${#UPDATE[@]}" -gt 0 ]; then printf '  %q' "${UPDATE[@]}"; echo; fi
printf '  %q' "${COMMAND[@]}"; echo
echo "GPU drivers and CUDA/ROCm SDKs are intentionally not changed."
if [ "$MODE" != "--execute" ]; then
    echo "Repeat with --execute after reviewing the commands."
    exit 0
fi
if [ "${#UPDATE[@]}" -gt 0 ]; then "${UPDATE[@]}"; fi
"${COMMAND[@]}"
