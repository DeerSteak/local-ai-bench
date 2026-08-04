"""Single source of truth for all model definitions. No external
dependencies, so this is safe to import before packages are installed."""

# "download_size" is rounded UP to the next 0.1 GB — see docs/workloads.md.
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

# "tier" maps onto the LLM tiers so --maxtier caps both together — see docs/cli-reference.md's `--maxtier` row.
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
        "resolutions": [(512, 512), (768, 768)],  # SD1.5's native range — see docs/workloads.md
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
        "tag":            "gemma3:1b-it-q4_K_M",
        "label":          "Gemma 3 1B",
        "short":          "gemma3-1b",
        "tier":           "xsmall",
        "download_size":  "~0.8 GB",
        "params_b":       1,
        "hf_repo":        "bartowski/google_gemma-3-1b-it-GGUF",
        "hf_file":        "google_gemma-3-1b-it-Q4_K_M.gguf",
    },
    {
        "tag":            "granite4.1:3b-q4_K_M",
        "label":          "Granite 4.1 3B Q4_K_M",
        "short":          "granite4.1-3b-q4",
        "tier":           "xsmall",
        "download_size":  "~2.1 GB",
        "params_b":       3,
        "hf_repo":        "ibm-granite/granite-4.1-3b-GGUF",
        "hf_file":        "granite-4.1-3b-Q4_K_M.gguf",
    },
    {
        "tag":            "qwen3.5:4b-q4_K_M",
        "label":          "Qwen3.5 4B Q4_K_M",
        "short":          "qwen3.5-4b-q4",
        "tier":           "xsmall",
        "download_size":  "~3.1 GB",
        "params_b":       4,
        "hf_repo":        "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "hf_file":        "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    },
], key=lambda m: m["params_b"])

# Small-tier models (≤20B parameters).
# "params_b" is total parameters (not active, for MoE models) and sets sort order below.
LLM_MODELS_SMALL = sorted([
    {
        "tag":            "granite4.1:8b-q4_K_M",
        "label":          "Granite 4.1 8B Q4_K_M",
        "short":          "granite4.1-8b-q4",
        "tier":           "small",
        "download_size":  "~5.4 GB",
        "params_b":       8,
        "hf_repo":        "ibm-granite/granite-4.1-8b-GGUF",
        "hf_file":        "granite-4.1-8b-Q4_K_M.gguf",
    },
    {
        "tag":            "qwen3.5:9b-q4_K_M",
        "label":          "Qwen3.5 9B Q4_K_M",
        "short":          "qwen3.5-9b-q4",
        "tier":           "small",
        "download_size":  "~6.2 GB",
        "params_b":       9,
        "hf_repo":        "bartowski/Qwen_Qwen3.5-9B-GGUF",
        "hf_file":        "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
    },
    {
        "tag":            "gemma4:12b-it-q4_K_M",
        "label":          "Gemma 4 12B Q4_K_M",
        "short":          "gemma4-12b-q4",
        "tier":           "small",
        "download_size":  "~7.7 GB",
        "params_b":       12,
        "hf_repo":        "bartowski/gemma-4-12B-it-GGUF",
        "hf_file":        "gemma-4-12B-it-Q4_K_M.gguf",
    },
], key=lambda m: m["params_b"])

# Medium-tier models (26-35B params) — one dense alongside two MoE entries; see docs/workloads.md#dense-vs-mixture-of-experts-moe.
LLM_MODELS_MEDIUM = sorted([
    {
        "tag":            "gemma3:27b-it-q4_K_M",
        "label":          "Gemma 3 27B Q4_K_M",
        "short":          "gemma3-27b-q4",
        "tier":           "medium",
        "download_size":  "~16.6 GB",
        "params_b":       27,
        "hf_repo":        "ggml-org/gemma-3-27b-it-GGUF",
        "hf_file":        "gemma-3-27b-it-Q4_K_M.gguf",
    },
    {
        "tag":            "nemotron-3-nano:30b-a3b-q4_K_M",
        "label":          "Nemotron 3 Nano 30B-A3B",
        "short":          "nemotron3-nano-30b-a3b",
        "tier":           "medium",
        "download_size":  "~24.0 GB",
        "params_b":       30,   # 3B active — hybrid Mamba-Transformer MoE
        "hf_repo":        "unsloth/Nemotron-3-Nano-30B-A3B-GGUF",
        "hf_file":        "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf",
    },
    {
        "tag":            "qwen3.6:35b-a3b",
        "label":          "Qwen3.6 35B-A3B",
        "short":          "qwen3.6-35b-a3b",
        "tier":           "medium",
        "download_size":  "~24.0 GB",
        "params_b":       35,   # 3B active
        "hf_repo":        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "hf_file":        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    },
], key=lambda m: m["params_b"])

# Large-tier models (70B+ params), same dense/MoE rationale as medium.
# Qwen3-Coder-Next and Nemotron 3 Super ship as multi-part GGUF splits (see docs/engines.md).
LLM_MODELS_LARGE = sorted([
    {
        "tag":            "llama3.3:70b-instruct-q4_K_M",
        "label":          "Llama 3.3 70B Q4_K_M",
        "short":          "llama3.3-70b-q4",
        "tier":           "large",
        "download_size":  "~39.7 GB",
        "params_b":       70,
        "hf_repo":        "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "hf_file":        "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    },
    {
        "tag":            "qwen3-coder-next:80b-a3b-q4_K_M",
        "label":          "Qwen3-Coder-Next 80B-A3B Q4_K_M",
        "short":          "qwen3-coder-next-80b-a3b-q4",
        "tier":           "large",
        "download_size":  "~48.4 GB",
        "params_b":       80,   # 3B active — hybrid attention MoE
        "hf_repo":        "Qwen/Qwen3-Coder-Next-GGUF",
        "hf_file":        [
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00002-of-00004.gguf",
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00003-of-00004.gguf",
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00004-of-00004.gguf",
        ],
    },
    {
        "tag":            "nemotron-3-super:120b",
        "label":          "Nemotron 3 Super 120B",
        "short":          "nemotron3-super-120b",
        "tier":           "large",
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
