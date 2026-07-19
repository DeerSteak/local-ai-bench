"""
config.py — shared constants for the benchmark suite.

Other modules should `import config` and reference `config.NAME` rather than
`from config import NAME` for any value main() can override via CLI flags
(currently RUN_TIMEOUT and ACCURACY_TEST_TIMEOUT) — a `from` import binds a
local copy at import time and won't see a later `config.RUN_TIMEOUT = ...`
assignment.
"""

from pathlib import Path

COMFYUI_URL  = "http://localhost:8188"

# llama-server's default port. LlamaCppEngine launches its own server on
# demand (one model per process), always on this fixed port.
LLAMACPP_PORT = 8080
LLAMACPP_URL  = f"http://localhost:{LLAMACPP_PORT}"

# Prompt-processing batch size, pinned on every request instead of left to the server's auto-detected default.
LLAMACPP_NUM_BATCH = 512

# Repo root — this file lives in scripts/, one level below it.
SCRIPT_DIR   = Path(__file__).resolve().parent.parent
COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"

# Vendored llama.cpp location (Linux source build / Windows prebuilt zip); macOS's brew
# install goes on PATH instead. LlamaCppEngine._binary_path checks both.
LLAMACPP_DIR = SCRIPT_DIR / "llama.cpp"

# Model downloads land here (setup_check.py), namespaced one subdirectory
# per engine (e.g. "llamacpp") so a future engine with its own model format/
# layout (e.g. MLX) doesn't collide with this one's.
MODELS_DIR = SCRIPT_DIR / "models"

RESULTS_DIR = SCRIPT_DIR / "results"

CONTEXT_LENGTHS = [2048, 8192, 32768, 65536]   # tokens (approximate, via prompt padding)

# Concurrency tests (scripts/concurrency_benchmark.py) — see docs/workloads.md.
# "tool" simulates agentic/tool-calling fan-out: a handful of concurrent
# requests, each a short tool-call-shaped turn — always runs every level, no
# soft-exit (see benchmark.py). "chat" simulates a chat server under load:
# many simultaneous long-conversation users, with a soft-exit once mean
# tok/s craters (CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT) so a model
# that's already clearly too slow doesn't burn huge wall-clock time climbing
# to 32-way for a foregone conclusion.
CONCURRENCY_TOOL_LEVELS  = [1, 2, 4, 6, 8, 12, 16]
CONCURRENCY_TOOL_CONTEXT = 4096    # tokens per concurrent request/slot
CONCURRENCY_CHAT_LEVELS  = [1, 2, 4, 8, 16, 24, 32]
CONCURRENCY_CHAT_CONTEXT = 16384   # tokens per concurrent request/slot
CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT = 8
IMAGE_RESOLUTIONS = [(1024, 1024), (1536, 1536)]
# Steps are per-model in IMAGE_MODELS
IMAGE_SEED  = 42
IMAGE_PROMPT = (
    "A photorealistic high-end gaming PC build with RGB lighting, "
    "multiple GPUs, custom water cooling, shot in a dark room, "
    "highly detailed, 8k resolution"
)

VERSION        = "3.0"
WARMUP_RUNS    = 2
N_RUNS         = 3   # measured runs per test — every test averages exactly this many
RUN_TIMEOUT = 300   # seconds per run (warmup and measured) before aborting — overridden by --timeout

# Per accuracy question (mcq/math/code), overridden by --acc-timeout — much shorter than RUN_TIMEOUT
# since a stuck model's unbounded token budget would otherwise burn 300s per question before anyone noticed.
ACC_TIMEOUT = 60

# How often the accuracy tests re-check a streaming response for a degenerate loop (Shared.looks_like_loop),
# rather than waiting the full ACC_TIMEOUT to look.
LOOP_CHECK_INTERVAL = 8

SLOW_MODEL_MIN_TPS = 15.0   # tokens/sec below which a model is skipped from the conversation test

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
