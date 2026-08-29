"""Shared constants. CLI-overridable ones (RUN_TIMEOUT, ACC_TIMEOUT, ACC_TOKEN_BUDGET, N_RUNS)
need `config.NAME` access — `from config import NAME` binds a stale copy before any override applies."""

import os
import math
from pathlib import Path

VERSION        = "6.0"
PREFILL_128K_TOKENS = 1 << 17

COMFYUI_URL  = "http://localhost:8188"

# llama-server's default port. LlamaCppEngine launches its own server on
# demand (one model per process), always on this fixed port.
LLAMACPP_PORT = 8080
LLAMACPP_URL  = f"http://localhost:{LLAMACPP_PORT}"

# Prompt-processing batch size, pinned on every request instead of left to the server's auto-detected default.
LLAMACPP_NUM_BATCH = 512

# KV cache quantization for llama-server — q8_0 halves KV cache memory vs f16 with negligible quality loss.
# Requires flash attention, which LlamaCppEngine.ensure_running passes alongside it.
LLAMACPP_KV_CACHE_TYPE = "q8_0"
LLAMACPP_GPU_SPLIT_MODE = "layer"
LLAMACPP_NO_REPACK = False
LLAMACPP_NO_HOST = False

# Repository root shared by all package groups.
SCRIPT_DIR   = Path(__file__).resolve().parents[2]
COMFYUI_DIR  = SCRIPT_DIR / "ComfyUI"
SETUP_CONFIG_PATH = SCRIPT_DIR / "local_ai_bench_config.json"

# Vendored llama.cpp location (Linux source build / Windows prebuilt zip); macOS's brew
# install goes on PATH instead. LlamaCppEngine._binary_path checks both.
LLAMACPP_DIR = SCRIPT_DIR / "llama.cpp"
LLAMACPP_VULKAN_DIR = SCRIPT_DIR / "llama.cpp-vulkan"

# Own venv: vLLM pins a torch build that would collide with bench-env's.
VLLM_VENV = SCRIPT_DIR / "vllm-env"
VLLM_PORT = 8000
VLLM_URL  = f"http://localhost:{VLLM_PORT}"
# vLLM preallocates this fraction of VRAM for weights + KV cache.
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLMBENCH_GPU_MEMORY_UTILIZATION = 0.85
VLLMBENCH_GB10_GPU_MEMORY_UTILIZATION = 0.10
VLLM_OFFLOAD_STEP_GB = 2
VLLM_OFFLOAD_RESERVE_GB = 3
VLLM_OFFLOAD_HOST_RESERVE_GB = 8
VLLM_OFFLOAD_MAX_ATTEMPTS = 4
# Prompts are padded by characters, so real tokenization can overshoot the target.
# vLLM rejects prompt+max_tokens > max_model_len outright; llama.cpp does not.
VLLM_CTX_TOLERANCE = 64

# vLLM's own default, then AMD `vllm-launch`'s.
VLLM_DISCOVERY_PORTS = (8000, 8001)

# Model downloads land here (setup_check.py), namespaced one subdirectory per engine — see docs/engines.md.
MODELS_DIR = SCRIPT_DIR / "models"
CUSTOM_MODELS_PATH = MODELS_DIR / "custom_models.json"
COMFYUI_MODELS_DIR = MODELS_DIR / "comfyui"
COMFYUI_EXTRA_MODEL_PATHS = COMFYUI_MODELS_DIR / "extra_model_paths.yaml"
RESUME_DIGEST_CACHE_PATH = SCRIPT_DIR / ".resume_digest_cache.json"

RESULTS_DIR = SCRIPT_DIR / "results"

CONTEXT_LENGTHS = [
    512, 2048, 8192, 32768, 65536, PREFILL_128K_TOKENS,
]   # tokens (approximate, via prompt padding)
ACCURACY_CONTEXT = 32768   # fixed llama-server allocation shared by accuracy warmup and questions

# See docs/workloads.md#concurrency for "tool" vs. "chat" and the soft-exit rationale.
CONCURRENCY_TOOL_LEVELS  = [1, 2, 4, 6, 8, 12, 16]
CONCURRENCY_TOOL_CONTEXT = 4096    # tokens per concurrent request/slot (padded prompt size)
CONCURRENCY_CHAT_LEVELS  = [1, 2, 4, 8, 16, 24, 32]
CONCURRENCY_CHAT_CONTEXT = 16384   # tokens per concurrent request/slot (padded prompt size)
CONCURRENCY_CHAT_MIN_LEVEL_BEFORE_SOFT_EXIT = 8

GENERATE_MAX_TOKENS = 512   # n_predict for engine.generate(); concurrency slot ctx must add this on top of the padded prompt
IMAGE_RESOLUTIONS = [(1024, 1024), (1536, 1536)]
# Steps are per-model in IMAGE_MODELS
IMAGE_SEED  = 42
IMAGE_PROMPT = (
    "A photorealistic high-end gaming PC build with RGB lighting, "
    "multiple GPUs, custom water cooling, shot in a dark room, "
    "highly detailed, 8k resolution"
)

WARMUP_RUNS    = 2
N_RUNS         = 3   # measured runs for single-shot LLM, embeddings, and images
RETRY_CRASHED_MODELS = False
RUN_TIMEOUT = 300   # base generation/chat timeout; images use 2x — overridden by --timeout

# Per accuracy question (mcq/math/reasoning/code/tool), overridden by --acc-timeout.
# A single shared deadline covers both bounded accuracy-generation passes.
ACC_TIMEOUT = 60
ACC_TOKEN_BUDGET = 8192
ACC_FINALIZE_FRACTION = 0.60
ACC_FINALIZE_MESSAGE = (
    "You reached the generation limit. Return a complete final answer now. "
    "Be concise and follow the requested answer format. Do not continue the "
    "previous fragment; provide the complete answer that should be graded."
)

# How often accuracy streams are checked for a degenerate generation loop.
# rather than waiting the full ACC_TIMEOUT to look.
LOOP_CHECK_INTERVAL = 8

SLOW_MODEL_MIN_TPS = 15.0   # tokens/sec below which a model is skipped from the conversation test

TELEMETRY_INTERVAL_SEC = float(os.environ.get("LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC", "0.5"))
if not math.isfinite(TELEMETRY_INTERVAL_SEC) or TELEMETRY_INTERVAL_SEC <= 0:
    raise ValueError("LOCAL_AI_BENCH_MEMORY_INTERVAL_SEC must be positive")
MEMORY_HEADROOM_COMFORTABLE_FRACTION = 0.20

SUSTAINED_DURATION_SEC = 600
SUSTAINED_WINDOW_SEC = 10
SUSTAINED_CONTEXT_TOKENS = 2048
SUSTAINED_MIN_CLASSIFICATION_SEC = 120
SUSTAINED_INITIAL_WINDOWS = 3
SUSTAINED_STEADY_WINDOWS = 6
SUSTAINED_ONSET_CONSECUTIVE_WINDOWS = 3
SUSTAINED_ONSET_TOLERANCE_FRACTION = 0.05
SUSTAINED_SIGNIFICANT_RETENTION = 0.85
SUSTAINED_MILD_RETENTION = 0.95
SUSTAINED_TEMPERATURE_RISE_C = 5.0
SUSTAINED_TEMPERATURE_CEILING_BAND_C = 2.0
SUSTAINED_POWER_DROP_FRACTION = 0.05

# Provisional product-relevance floors; qualification may only raise these above observed noise.
PRACTICAL_TTFT_THRESHOLD_PCT = 8.0
PRACTICAL_THROUGHPUT_THRESHOLD_PCT = 3.0
PRACTICAL_WALL_TIME_THRESHOLD_PCT = 3.0
PRACTICAL_ACCURACY_THRESHOLD_PCT = 1.0

# llama-bench pp/tg throughput sweep (opt-in `llamabench` test) — see docs/workloads.md#llama-bench.
# Matches every non-zero size from CONTEXT_LENGTHS (prefill) and LLMConversationBenchmark.CONV_CHECKPOINTS
# (conversation) so llama-bench numbers can stand in for both as they're phased out.
LLAMABENCH_PP = [
    512, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 81920, 98304,
    PREFILL_128K_TOKENS,
]
LLAMABENCH_TG = [128, 512]
LLAMABENCH_BATCH_SIZE = 2048
LLAMABENCH_UBATCH_SIZE = 512
# Idle timeout for one whole pp/tg sweep — killed if no stdout/stderr line arrives for this
# long, not a ceiling on total sweep time (a full sweep can legitimately run for hours).
# Independent of --timeout, like embeddings' own fixed 120s.
LLAMABENCH_TIMEOUT = 1800
# Exceeds any real model's layer count so every layer offloads — llama-bench's own
# default (-1) isn't documented as meaning "all layers".
LLAMABENCH_FULL_OFFLOAD_NGL = 999

# `vllm bench` latency/throughput sweep (opt-in `vllmbench` test) — see docs/workloads.md#vllm-bench.
# Uses LLAMABENCH_PP for input shapes so the shared prompt cap applies identically.
VLLMBENCH_OUTPUT = [128, 512]
VLLMBENCH_BATCH_SIZE = 1
# vllm bench latency defaults to 30 iterations and 10 warmups, far more than this suite needs.
VLLMBENCH_ITERS = 3
VLLMBENCH_WARMUP_ITERS = 1
VLLMBENCH_NUM_PROMPTS = 32
VLLMBENCH_TIMEOUT = 1800
VLLM_COLD_IMPORT_TIMEOUT = 300

# llama-batched-bench concurrency sweep (opt-in `llamabenchconc` test) — see docs/workloads.md#llama-bench-concurrency.
LLAMABENCH_CONC_PP = 4096   # matches CONCURRENCY_TOOL_CONTEXT, so this cross-checks conc_tool at the same depth
LLAMABENCH_CONC_TG = [128, 512]
LLAMABENCH_CONC_NPL = [1, 2, 4, 8, 16]
LLAMABENCH_CONC_GPU_LAYERS = "auto"

# Above this, a self-reported tps is treated as unreliable — see docs/engines.md's "_sanitize_tps".
MAX_PLAUSIBLE_TPS = 5000.0

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
