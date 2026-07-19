"""
models.py — Single source of truth for all model definitions.

No external dependencies: safe to import before packages are installed.
Both benchmark.py and setup_check.py import from here.
"""

# "download_size" is each model's on-disk size, rounded UP to the next 0.1 GB
# so setup's disk-space check errs toward requiring more free space, not less.
# "hf_repo"/"hf_file" locate the GGUF on HuggingFace for setup_check.py's downloader;
# "hf_file" is a list for models split across multiple GGUF parts.
EMBED_MODELS = [
    {
        "tag":            "nomic-embed-text",
        "label":          "Nomic Embed Text",
        "short":          "nomic-embed-text",
        "download_size":  "~0.3 GB",
        "hf_repo":        "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "hf_file":        "nomic-embed-text-v1.5.f16.gguf",
    },
    {
        "tag":            "mxbai-embed-large",
        "label":          "MixedBread Embed Large",
        "short":          "mxbai-embed-large",
        "download_size":  "~0.7 GB",
        "hf_repo":        "ChristianAzinn/mxbai-embed-large-v1-gguf",
        "hf_file":        "mxbai-embed-large-v1_fp16.gguf",
    },
]

# Checkpoints missing from ComfyUI/models/checkpoints/ are skipped automatically.
# "tier" maps each checkpoint onto the LLM tiers' xsmall/small/medium/large
# scale so --maxtier caps both together.
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
LLM_MODELS_XSMALL = sorted([
    {
        "tag":            "llama3.2:3b-instruct-q4_K_M",
        "label":          "Llama 3.2 3B Q4_K_M",
        "short":          "llama3.2-3b-q4",
        "download_size":  "~2.1 GB",
        "params_b":       3,
        "hf_repo":        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "hf_file":        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    },
    {
        "tag":            "phi4-mini",
        "label":          "Phi 4 Mini",
        "short":          "phi4-mini",
        "download_size":  "~2.5 GB",
        "params_b":       3.8,
        "hf_repo":        "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "hf_file":        "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
    },
], key=lambda m: m["params_b"])

# Small-tier models (≤20B parameters).
# "params_b" is total parameters (not active, for MoE models) and sets sort order below.
LLM_MODELS_SMALL = sorted([
    {
        "tag":            "mistral:7b-instruct-v0.3-q4_K_M",
        "label":          "Mistral 7B v0.3 Q4_K_M",
        "short":          "mistral-7b-q4",
        "download_size":  "~4.4 GB",
        "params_b":       7,
        "hf_repo":        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "hf_file":        "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    },
    {
        "tag":            "llama3.1:8b-instruct-q4_K_M",
        "label":          "Llama 3.1 8B Q4_K_M",
        "short":          "llama3.1-8b-q4",
        "download_size":  "~5.0 GB",
        "params_b":       8,
        "hf_repo":        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "hf_file":        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    },
], key=lambda m: m["params_b"])

# Medium-tier models (26–35B parameters).
LLM_MODELS_MEDIUM = sorted([
    {
        "tag":            "nemotron-3-nano:30b-a3b-q4_K_M",
        "label":          "Nemotron 3 Nano 30B-A3B",
        "short":          "nemotron3-nano-30b-a3b",
        "download_size":  "~24.0 GB",
        "params_b":       30,   # 3B active — hybrid Mamba-Transformer MoE
        "hf_repo":        "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        "hf_file":        "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
    },
    {
        "tag":            "qwen3.6:35b-a3b",
        "label":          "Qwen3.6 35B-A3B",
        "short":          "qwen3.6-35b-a3b",
        "download_size":  "~24.0 GB",
        "params_b":       35,   # 3B active
        "hf_repo":        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "hf_file":        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    },
], key=lambda m: m["params_b"])

# Large-tier models (70B+ parameters). Llama 4 Scout and Nemotron 3 Super ship
# as multi-part GGUF splits — "hf_file" is a list, part 1 first; llama.cpp
# auto-discovers the sibling parts next to it, so only the first path is ever
# passed to -m.
LLM_MODELS_LARGE = sorted([
    {
        "tag":            "llama4:16x17b",
        "label":          "Llama 4 Scout 16x17B",
        "short":          "llama4-16x17b",
        "download_size":  "~67.0 GB",
        "params_b":       109,   # 17B active
        "hf_repo":        "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
        "hf_file":        [
            "UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00001-of-00002.gguf",
            "UD-Q4_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q4_K_XL-00002-of-00002.gguf",
        ],
    },
    {
        "tag":            "nemotron-3-super:120b",
        "label":          "Nemotron 3 Super 120B",
        "short":          "nemotron3-super-120b",
        "download_size":  "~87.0 GB",
        "params_b":       120,   # 12B active — hybrid Mamba-Transformer MoE
        "hf_repo":        "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
        "hf_file":        [
            "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf",
            "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00002-of-00003.gguf",
            "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00003-of-00003.gguf",
        ],
    },
], key=lambda m: m["params_b"])

LLM_MODELS = LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE
