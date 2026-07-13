"""
models.py — Single source of truth for all model definitions.

No external dependencies: safe to import before packages are installed.
Both benchmark.py and setup_check.py import from here.
"""

# "download_size" is each model's on-disk size, rounded UP to the next 0.1 GB
# so setup's disk-space check errs toward requiring more free space, not less.
EMBED_MODELS = [
    {
        "tag":            "nomic-embed-text",
        "label":          "Nomic Embed Text",
        "short":          "nomic-embed-text",
        "download_size":  "~0.3 GB",
    },
    {
        "tag":            "mxbai-embed-large",
        "label":          "MixedBread Embed Large",
        "short":          "mxbai-embed-large",
        "download_size":  "~0.7 GB",
    },
]

# Image generation models. Checkpoint files not present in
# ComfyUI/models/checkpoints/ are skipped automatically.
# "tier" maps each checkpoint onto the same xsmall/small/medium/large scale as
# the LLM tiers (roughly by checkpoint size) so --maxtier caps both together.
IMAGE_MODELS = [
    {
        "label":       "Stable Diffusion 1.5",
        "checkpoint":  "v1-5-pruned-emaonly.safetensors",
        "workflow":    "sdxl",  # same minimal loader→CLIP→KSampler→VAE graph works unchanged
        "steps":       20,
        "cfg":         7.5,
        "sampler":     "euler",
        "scheduler":   "normal",
        "short":       "sd15",
        "tier":        "xsmall",  # ~4.3 GB
        # SD1.5 was trained at 512x512; the default 1024/1536 resolutions push
        # it far outside that range and produce degraded (duplicated-subject)
        # output, so it gets its own native-range pair instead.
        "resolutions": [(512, 512), (768, 768)],
    },
    {
        "label":      "SDXL",
        "checkpoint": "sd_xl_base_1.0.safetensors",
        "workflow":   "sdxl",
        "steps":      20,
        "cfg":        7.0,
        "sampler":    "euler_ancestral",
        "scheduler":  "normal",
        "short":      "sdxl",
        "tier":       "small",     # ~7.0 GB
    },
    {
        "label":      "SD3.5 Large",
        "checkpoint": "sd3.5_large.safetensors",
        "workflow":   "sd3",
        "steps":      28,
        "cfg":        4.5,
        "sampler":    "euler",
        "scheduler":  "beta",
        "short":      "sd35-large",
        "tier":       "medium",    # ~16.5 GB
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
        "tier":       "large",     # ~23.9 GB
    },
    {
        "label":      "Flux.2-dev",
        "checkpoint": "flux2-dev.safetensors",
        "workflow":   "flux2",
        "steps":      28,
        "cfg":        4.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux2-dev",
        "tier":       "large",     # ~64.5 GB
    },
]

# Extra-small-tier models (<6B parameters).
# Tags verified against ollama.com/library June 2026.
LLM_MODELS_XSMALL = sorted([
    {
        "tag":            "llama3.2:3b-instruct-q4_K_M",
        "label":          "Llama 3.2 3B Q4_K_M",
        "short":          "llama3.2-3b-q4",
        "download_size":  "~2.1 GB",
        "params_b":       3,
    },
    {
        "tag":            "phi4-mini",
        "label":          "Phi 4 Mini",
        "short":          "phi4-mini",
        "download_size":  "~2.5 GB",
        "params_b":       3.8,
    },
    {
        "tag":            "qwen3.5:4b",
        "label":          "Qwen3.5 4B",
        "short":          "qwen3.5-4b",
        "download_size":  "~3.4 GB",
        "params_b":       4,
    },
], key=lambda m: m["params_b"])

# Small-tier models (≤20B parameters).
# Tags verified against ollama.com/library June 2026.
# "params_b" is total parameter count in billions (not active/effective count
# for MoE models — e.g. Qwen3.6 35B-A3B has 3B active but 35B total) and sets
# the test order below, not download size or list position.
LLM_MODELS_SMALL = sorted([
    {
        "tag":            "llama3.1:8b-instruct-q4_K_M",
        "label":          "Llama 3.1 8B Q4_K_M",
        "short":          "llama3.1-8b-q4",
        "download_size":  "~5.0 GB",
        "params_b":       8,
    },
    {
        "tag":            "gemma4:e4b",
        "label":          "Gemma 4 E4B",
        "short":          "gemma4-e4b",
        "download_size":  "~9.7 GB",
        "params_b":       8,   # "E4B" = 4B effective; ~8B total raw parameters
    },
    {
        "tag":            "gpt-oss:20b",
        "label":          "GPT-OSS 20B (MXFP4)",
        "short":          "gpt-oss-20b",
        "download_size":  "~13.8 GB",
        "params_b":       20,
    },
], key=lambda m: m["params_b"])

# Medium-tier models (26–35B parameters).
LLM_MODELS_MEDIUM = sorted([
    {
        "tag":            "gemma4:26b",
        "label":          "Gemma 4 26B",
        "short":          "gemma4-26b",
        "download_size":  "~18.0 GB",
        "params_b":       26,
    },
    {
        "tag":            "deepseek-r1:32b",
        "label":          "DeepSeek-R1 32B",
        "short":          "deepseek-r1-32b",
        "download_size":  "~19.9 GB",
        "params_b":       32,
    },
    {
        "tag":            "qwen3.6:35b-a3b",
        "label":          "Qwen3.6 35B-A3B",
        "short":          "qwen3.6-35b-a3b",
        "download_size":  "~24.0 GB",
        "params_b":       35,   # 3B active
    },
], key=lambda m: m["params_b"])

# Large-tier models (70B+ parameters).
LLM_MODELS_LARGE = sorted([
    {
        "tag":            "llama3.3:70b-instruct-q4_K_M",
        "label":          "Llama 3.3 70B Q4_K_M",
        "short":          "llama3.3-70b-q4",
        "download_size":  "~42.6 GB",
        "params_b":       70,
    },
    {
        "tag":            "deepseek-r1:70b",
        "label":          "DeepSeek-R1 70B",
        "short":          "deepseek-r1-70b",
        "download_size":  "~42.6 GB",
        "params_b":       70,
    },
    {
        "tag":            "gpt-oss:120b",
        "label":          "GPT-OSS 120B (MXFP4)",
        "short":          "gpt-oss-120b",
        "download_size":  "~65.4 GB",
        "params_b":       120,
    },
], key=lambda m: m["params_b"])

LLM_MODELS = LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE
