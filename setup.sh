#!/usr/bin/env bash
# setup.sh — local-ai-bench setup for macOS and Linux
# Usage: bash setup.sh
set -euo pipefail

VENV_DIR="bench-env"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=11

# ── Colors ─────────────────────────────────────────────────────────────────────
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; CYAN="\033[96m"; BOLD="\033[1m"; RESET="\033[0m"
ok()      { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn()    { echo -e "  ${YELLOW}!${RESET}  $*"; }
fail()    { echo -e "  ${RED}✗${RESET}  $*"; }
info()    { echo -e "  ${CYAN}→${RESET}  $*"; }
section() { echo -e "\n${BOLD}──────────────────────────────────────────────────\n  $*\n──────────────────────────────────────────────────${RESET}"; }

OS="$(uname -s)"

# ── 1. Find or install Python 3.11+ ───────────────────────────────────────────
section "Python"

find_python() {
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local major minor
            major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo 0)
            minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo 0)
            if [ "$major" -ge "$PYTHON_MIN_MAJOR" ] && [ "$minor" -ge "$PYTHON_MIN_MINOR" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=""
if PYTHON=$(find_python); then
    ok "Found $($PYTHON --version) at $(command -v $PYTHON)"
else
    warn "Python $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR+ not found — installing..."
    if [ "$OS" = "Darwin" ]; then
        if ! command -v brew &>/dev/null; then
            info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            # Add brew to PATH for this session
            eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
        fi
        brew install python@3.11
        PYTHON=/opt/homebrew/bin/python3.11
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        PYTHON=python3.11
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3.11
        PYTHON=python3.11
    elif command -v snap &>/dev/null; then
        sudo snap install python311
        PYTHON=python3.11
    else
        fail "Could not install Python automatically. Please install Python 3.11+ manually and re-run."
        exit 1
    fi
    ok "Installed $($PYTHON --version)"
fi

# ── 2. Create venv ─────────────────────────────────────────────────────────────
section "Virtual Environment"

if [ -d "$VENV_DIR" ]; then
    ok "Venv already exists at $VENV_DIR"
else
    info "Creating venv at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Venv created"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ── 3. Base Python dependencies ────────────────────────────────────────────────
section "Python Packages"
info "Installing from requirements.txt ..."
"$VENV_PIP" install -r requirements.txt
ok "Base dependencies installed"

# ── 4. DGX Spark / Linux: install Ollama if missing ───────────────────────────
if [ "$OS" = "Linux" ] && ! command -v ollama &>/dev/null; then
    section "Ollama"
    warn "Ollama not found — installing via snap..."
    if command -v snap &>/dev/null; then
        sudo snap install ollama
        ok "Ollama installed via snap"
    else
        info "Installing Ollama via install script..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Ollama installed"
    fi
fi

# ── 4. Run setup_check.py inside the venv ─────────────────────────────────────
section "Running setup_check.py"
info "Using $($VENV_PYTHON --version) from $VENV_PYTHON"

"$VENV_PYTHON" setup_check.py

# ── 5. Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Setup complete.${RESET}"
echo ""
echo -e "  To run benchmarks:"
echo -e "    ${CYAN}bash run_linux.sh${RESET}"
echo ""
read -r -p "  Run the benchmark now? [y/N] " _reply
echo ""
if [[ "$_reply" =~ ^[Yy](es)?$ ]]; then
    bash "$(dirname "$0")/run_linux.sh"
fi
