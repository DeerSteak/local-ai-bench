#!/usr/bin/env bash
# setup.sh — local-ai-bench setup for macOS and Linux
# Usage: bash setup.sh
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_ROOT"

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
    warn "Python $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR+ not found"

    NEED_BREW=0
    if [ "$OS" = "Darwin" ] && ! command -v brew &>/dev/null; then
        NEED_BREW=1
    fi

    echo ""
    echo "  This will:"
    if [ "$NEED_BREW" = "1" ]; then
        echo "    • Install Homebrew"
    fi
    if [ "$OS" = "Darwin" ]; then
        echo "    • Install Python 3.11 via Homebrew"
    elif command -v apt-get &>/dev/null; then
        echo "    • Install python3.11, its venv module, and development headers via apt-get (requires sudo)"
    elif command -v dnf &>/dev/null; then
        echo "    • Install python3.11 and its development headers via dnf (requires sudo)"
    elif command -v snap &>/dev/null; then
        echo "    • Install python311 via snap (requires sudo)"
    else
        fail "Could not install Python automatically. Please install Python 3.11+ manually and re-run."
        exit 1
    fi
    echo ""
    read -r -p "  Proceed? [Y/n] " _py_reply || _py_reply="y"
    echo ""
    if [[ -n "$_py_reply" && ! "$_py_reply" =~ ^[Yy] ]]; then
        fail "Setup cancelled — Python 3.11+ is required."
        exit 1
    fi

    info "Installing..."
    if [ "$OS" = "Darwin" ]; then
        if [ "$NEED_BREW" = "1" ]; then
            info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            # Add brew to PATH for this session
            eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
        fi
        brew install python@3.11
        PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        PYTHON=python3.11
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3.11 python3.11-devel
        PYTHON=python3.11
    elif command -v snap &>/dev/null; then
        sudo snap install python311
        PYTHON=python3.11
    fi
    ok "Installed $($PYTHON --version)"
fi

# Triton (a vLLM dependency) compiles a CUDA helper at import time and needs Python.h.
# Installed here, with any other prerequisite, so vLLM never triggers a later sudo prompt.
# 3.12 is vllm_install.PINNED_PYTHON, the interpreter its ROCm/DGX-Spark wheels require.
if [ "$OS" = "Linux" ]; then
    PYTHON_SERIES=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MISSING_HEADERS=()
    for _series in $(printf '%s\n' "$PYTHON_SERIES" 3.12 | sort -u); do
        if [ ! -f "/usr/include/python${_series}/Python.h" ]; then
            MISSING_HEADERS+=("$_series")
        fi
    done
    if [ ${#MISSING_HEADERS[@]} -gt 0 ]; then
        if command -v apt-get &>/dev/null; then
            HEADER_PACKAGES=()
            for _series in "${MISSING_HEADERS[@]}"; do
                if apt-cache show "python${_series}-dev" &>/dev/null; then
                    HEADER_PACKAGES+=("python${_series}-dev")
                fi
            done
            if [ ${#HEADER_PACKAGES[@]} -gt 0 ]; then
                info "Installing Python development headers (${HEADER_PACKAGES[*]}) — needed by vLLM ..."
                sudo apt-get install -y "${HEADER_PACKAGES[@]}" || \
                    warn "Header install failed — vLLM will not start until it succeeds"
            fi
        elif command -v dnf &>/dev/null; then
            info "Installing Python development headers — needed by vLLM ..."
            sudo dnf install -y python3-devel || \
                warn "Header install failed — vLLM will not start until it succeeds"
        fi
    fi

    # Debian/Ubuntu ship ensurepip separately, and without it `python -m venv` fails
    # for an interpreter this script found rather than installed.
    if ! "$PYTHON" -c "import ensurepip" &>/dev/null; then
        if command -v apt-get &>/dev/null && apt-cache show "python${PYTHON_SERIES}-venv" &>/dev/null; then
            info "Installing the Python venv module (python${PYTHON_SERIES}-venv)..."
            sudo apt-get install -y "python${PYTHON_SERIES}-venv" || \
                warn "venv install failed — creating $VENV_DIR will not succeed"
        else
            warn "Python's venv module is missing and no package was found to install it."
        fi
    fi

    # llama.cpp builds from source on Linux; discovering a missing tool mid-install would
    # strand the run after its last approval prompt.
    if ! command -v llama-cli &>/dev/null && ! command -v llama-server &>/dev/null; then
        BUILD_TOOLS=()
        for _tool in git cmake; do
            command -v "$_tool" &>/dev/null || BUILD_TOOLS+=("$_tool")
        done
        if [ ${#BUILD_TOOLS[@]} -gt 0 ]; then
            if command -v apt-get &>/dev/null; then
                info "Installing llama.cpp build prerequisites (${BUILD_TOOLS[*]} build-essential) ..."
                sudo apt-get install -y "${BUILD_TOOLS[@]}" build-essential || \
                    warn "Install failed — llama.cpp cannot build without ${BUILD_TOOLS[*]}"
            elif command -v dnf &>/dev/null; then
                info "Installing llama.cpp build prerequisites (${BUILD_TOOLS[*]}) ..."
                sudo dnf install -y "${BUILD_TOOLS[@]}" gcc-c++ make || \
                    warn "Install failed — llama.cpp cannot build without ${BUILD_TOOLS[*]}"
            else
                warn "llama.cpp needs ${BUILD_TOOLS[*]} to build — install them and re-run"
            fi
        fi
    fi
fi

GUI_SESSION=0
if [ -z "${SSH_CONNECTION:-}${SSH_CLIENT:-}${SSH_TTY:-}" ]; then
    if [ "$OS" = "Darwin" ] || [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        GUI_SESSION=1
    fi
fi
if [ "$GUI_SESSION" = "1" ] && ! "$PYTHON" -c "import tkinter" >/dev/null 2>&1; then
    warn "Tkinter is not available, so the graphical setup wizard cannot start."
    read -r -p "  Install Tkinter support? [Y/n] " _tk_reply || _tk_reply="y"
    if [[ -z "$_tk_reply" || "$_tk_reply" =~ ^[Yy] ]]; then
        if [ "$OS" = "Darwin" ] && command -v brew &>/dev/null; then
            PYTHON_SERIES=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            brew install "python-tk@$PYTHON_SERIES"
        elif command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y python3-tk
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3-tkinter
        else
            warn "Install Tkinter with your package manager to use the GUI; continuing with terminal setup."
        fi
        if "$PYTHON" -c "import tkinter" >/dev/null 2>&1; then
            ok "Tkinter support available"
        else
            warn "Tkinter is still unavailable; continuing with terminal setup."
        fi
    else
        info "Continuing with terminal setup"
    fi
fi

# ── 2. Create venv ─────────────────────────────────────────────────────────────
section "Virtual Environment"

if [ -x "$VENV_DIR/bin/python" ]; then
    if ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
        fail "$VENV_DIR uses Python older than 3.11. Move or remove it, then re-run setup."
        exit 1
    fi
    # venv symlinks bin/python before running ensurepip, so pip is the real completeness test.
    if [ ! -x "$VENV_DIR/bin/pip" ]; then
        fail "$VENV_DIR is missing pip — an earlier setup run failed partway. Remove it, then re-run setup."
        exit 1
    fi
    ok "Venv already exists at $VENV_DIR"
elif [ -e "$VENV_DIR" ]; then
    fail "$VENV_DIR exists but is not a usable virtual environment. Move or remove it, then re-run setup."
    exit 1
else
    info "Creating venv at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Venv created"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

# ── 3. Base Python dependencies ────────────────────────────────────────────────
# ── 4. Run setup_check.py inside the venv ─────────────────────────────────────
# (llama.cpp detection/install — including on Linux — happens inside
# setup_check.py, gated behind its own approval prompt, so it isn't
# installed here without asking.)
section "Running setup_check.py"
info "Using $($VENV_PYTHON --version) from $VENV_PYTHON"

"$VENV_PYTHON" -m scripts.setup.setup_check "$@"

# ── 5. Done ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Setup complete.${RESET}"
echo ""
echo -e "  To run benchmarks:"
echo -e "    ${CYAN}bash run_bench.sh${RESET}"
echo ""
read -r -p "  Run the benchmark now? [y/N] " _reply
echo ""
if [[ "$_reply" =~ ^[Yy](es)?$ ]]; then
    bash "$(dirname "$0")/run_bench.sh"
fi
