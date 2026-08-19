#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SYSTEM="$(uname -s)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/qualification_python.sh"
if [ "$SYSTEM" = "Darwin" ]; then
    if qualification_python >/dev/null && command -v git >/dev/null 2>&1; then
        echo "Qualification host prerequisites are already installed."
        exit 0
    fi
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew is required to install Python unattended; install it from https://brew.sh and rerun." >&2
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
    COMMAND=(sudo -n apt-get install -y git cmake build-essential curl libopenmpi-dev python3 python3-venv python3-dev)
    UPDATE=(sudo -n apt-get update)
elif command -v dnf >/dev/null 2>&1; then
    COMMAND=(sudo -n dnf install -y git cmake gcc-c++ make curl openmpi openmpi-devel python3 python3-devel)
    UPDATE=()
else
    echo "Unsupported host package manager; install Git, CMake, a C++ compiler, curl, OpenMPI, and Python 3.11+." >&2
    exit 1
fi

echo "Qualification host bootstrap preview:"
if [ "${#UPDATE[@]}" -gt 0 ]; then printf '  %q' "${UPDATE[@]}"; echo; fi
printf '  %q' "${COMMAND[@]}"; echo
if ! qualification_python_312 >/dev/null; then
    echo "  install uv and a private CPython 3.12 for isolated vLLM qualification"
fi
echo "GPU drivers and CUDA/ROCm SDKs are intentionally not changed."
if [ "$MODE" != "--execute" ]; then
    echo "Repeat with --execute after reviewing the commands."
    exit 0
fi
if [ "${#UPDATE[@]}" -gt 0 ]; then "${UPDATE[@]}"; fi
"${COMMAND[@]}"
if ! qualification_python_312 >/dev/null; then
    UV_BIN="$(command -v uv || true)"
    if [ -z "$UV_BIN" ]; then
        UV_INSTALLER="$(mktemp)"
        curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$UV_INSTALLER"
        sh "$UV_INSTALLER"
        rm -f "$UV_INSTALLER"
        UV_BIN="${HOME}/.local/bin/uv"
    fi
    "$UV_BIN" python install 3.12
fi
