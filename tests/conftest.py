"""conftest.py — makes scripts/ importable as top-level modules (config, shared,
models, ...) since that's how the scripts themselves import each other."""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
