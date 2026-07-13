"""
config.py — shared constants for the benchmark suite.

Other modules should `import config` and reference `config.NAME` rather than
`from config import NAME` for any value main() can override via CLI flags
(currently just RUN_TIMEOUT) — a `from` import binds a local copy at import
time and won't see a later `config.RUN_TIMEOUT = ...` assignment.
"""

from pathlib import Path

OLLAMA_URL   = "http://localhost:11434"
COMFYUI_URL  = "http://localhost:8188"

# Repo root — this file lives in scripts/, one level below it.
SCRIPT_DIR   = Path(__file__).resolve().parent.parent
COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"

# Benchmark output — results JSON plus generated images. Each run's images
# land in results/images_<hostname>_<timestamp>/, a sibling of the matching
# results_<hostname>_<timestamp>.json, so both sort together by the shared
# hostname+timestamp.
RESULTS_DIR = SCRIPT_DIR / "results"

CONTEXT_LENGTHS = [2048, 8192, 32768, 65536]   # tokens (approximate, via prompt padding)
IMAGE_RESOLUTIONS = [(1024, 1024), (1536, 1536)]
# Steps are per-model in IMAGE_MODELS
IMAGE_SEED  = 42
IMAGE_PROMPT = (
    "A photorealistic high-end gaming PC build with RGB lighting, "
    "multiple GPUs, custom water cooling, shot in a dark room, "
    "highly detailed, 8k resolution"
)

VERSION        = "2.0"
WARMUP_RUNS    = 2
N_RUNS         = 3   # measured runs per test — every test averages exactly this many
RUN_TIMEOUT = 300   # seconds per run (warmup and measured) before aborting — overridden by --timeout

# Tokens/sec below which a model is skipped from the (expensive) conversation
# test — too slow for usable back-and-forth chat regardless of single-shot
# TTFT. Checked against every context depth the single-shot LLM test reported,
# not just one.
SLOW_MODEL_MIN_TPS = 15.0   # tokens/sec

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
