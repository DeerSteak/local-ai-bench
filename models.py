"""
models.py — Single source of truth for all model definitions.

No external dependencies: safe to import before packages are installed.
Both benchmark.py and setup_check.py import from here.
"""

EMBED_MODEL = "mxbai-embed-large"

# Image generation models. Checkpoint files not present in
# ComfyUI/models/checkpoints/ are skipped automatically.
IMAGE_MODELS = [
    {
        "label":      "SDXL",
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "workflow":   "sdxl",
        "steps":      20,
        "cfg":        7.0,
        "sampler":    "euler_ancestral",
        "scheduler":  "normal",
        "short":      "sdxl",
    },
    {
        "label":      "Flux.1-schnell",
        "checkpoint": "flux1-schnell.safetensors",
        "workflow":   "flux",
        "steps":      4,
        "cfg":        1.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux-schnell",
    },
    {
        "label":      "Flux.1-dev",
        "checkpoint": "flux1-dev.safetensors",
        "workflow":   "flux",
        "steps":      20,
        "cfg":        1.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux-dev",
    },
]

# Small-tier models (≤16GB VRAM) — run on all hardware including 8GB GPUs.
# Tags verified against ollama.com/library June 2026.
LLM_MODELS_SMALL = [
    {
        "tag":   "llama3.1:8b-instruct-q3_K_M",
        "label": "Llama 3.1 8B Q3_K_M",
        "short": "llama3.1-8b-q3",
        "vram":  "~4.3 GB",
    },
    {
        "tag":   "llama3.1:8b-instruct-q4_K_M",
        "label": "Llama 3.1 8B Q4_K_M",
        "short": "llama3.1-8b-q4",
        "vram":  "~4.9 GB",
    },
    {
        "tag":     "qwen3:14b-q4_K_M",
        "label":   "Qwen3 14B Q4_K_M",
        "short":   "qwen3-14b-q4",
        "vram":    "~9.3 GB",
        "max_ctx": 32768,
    },
    {
        "tag":   "gpt-oss:20b",
        "label": "GPT-OSS 20B (MXFP4)",
        "short": "gpt-oss-20b",
        "vram":  "~14 GB",
    },
]

# Medium-tier models (16–32GB VRAM) — 24 GB GPUs (RTX 3090/4090) and 32 GB unified memory.
LLM_MODELS_MEDIUM = [
    {
        "tag":     "qwen3:14b-q8_0",
        "label":   "Qwen3 14B Q8_0",
        "short":   "qwen3-14b-q8",
        "vram":    "~16 GB",
        "max_ctx": 32768,
    },
    {
        "tag":   "qwen3.6:35b-a3b",
        "label": "Qwen3.6 35B-A3B",
        "short": "qwen3.6-35b-a3b",
        "vram":  "~22 GB",
    },
]

# Large-tier models (≥32GB VRAM).
# Note: gpt-oss:120b ships in MXFP4 only — no Q3/Q4 variants exist.
LLM_MODELS_LARGE = [
    {
        "tag":   "llama3.1:70b-instruct-q3_K_M",
        "label": "Llama 3.1 70B Q3_K_M",
        "short": "llama3.1-70b-q3",
        "vram":  "~32 GB",
    },
    {
        "tag":   "llama3.1:70b-instruct-q4_K_M",
        "label": "Llama 3.1 70B Q4_K_M",
        "short": "llama3.1-70b-q4",
        "vram":  "~42 GB",
    },
    {
        "tag":   "gpt-oss:120b",
        "label": "GPT-OSS 120B (MXFP4)",
        "short": "gpt-oss-120b",
        "vram":  "~65 GB",
    },
]

LLM_MODELS = LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE
