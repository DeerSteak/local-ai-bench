#!/usr/bin/env bash

qualification_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3.14 python3.13 python3.12 python3.11 python3; do
        if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 \
                && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
                    >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

qualification_python_312() {
    local candidate
    for candidate in python3.12 "${HOME}/.local/bin/python3.12"; do
        if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' \
                    >/dev/null 2>&1; then
                command -v "$candidate" 2>/dev/null || printf '%s\n' "$candidate"
                return 0
            fi
        fi
    done
    return 1
}
