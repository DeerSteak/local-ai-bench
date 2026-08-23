"""Single source of truth for all model definitions. No external
dependencies, so this is safe to import before packages are installed."""

from pathlib import Path

# "download_size" is rounded UP to the next 0.1 GB — see docs/workloads.md.
EMBED_MODELS = [
    {
        "tag":            "nomic-embed-text",
        "label":          "Nomic Embed Text",
        "short":          "nomic-embed-text",
        "download_size":  "~0.3 GB",
        "hf_repo":        "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "hf_file":        "nomic-embed-text-v1.5.f16.gguf",
        "vllm_repo":      "nomic-ai/nomic-embed-text-v1.5",
        "vllm_download_size": "~0.6 GB",
    },
    {
        "tag":            "mxbai-embed-large",
        "label":          "MixedBread Embed Large",
        "short":          "mxbai-embed-large",
        "download_size":  "~0.7 GB",
        "hf_repo":        "ChristianAzinn/mxbai-embed-large-v1-gguf",
        "hf_file":        "mxbai-embed-large-v1_fp16.gguf",
        "vllm_repo":      "mixedbread-ai/mxbai-embed-large-v1",
        "vllm_download_size": "~0.7 GB",
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
        "label":      "Z-Image Turbo",
        "checkpoint": "z_image_turbo_bf16.safetensors",
        "checkpoint_folder": "diffusion_models",
        "checkpoint_loader": "UNETLoader",
        "checkpoint_repo": "Comfy-Org/z_image_turbo",
        "checkpoint_remote": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "support_assets": [
            {
                "name": "qwen_3_4b.safetensors",
                "folder": "text_encoders",
                "repo": "Comfy-Org/z_image_turbo",
                "remote": "split_files/text_encoders/qwen_3_4b.safetensors",
            },
            {
                "name": "z_image_ae.safetensors",
                "folder": "vae",
                "repo": "Comfy-Org/z_image_turbo",
                "remote": "split_files/vae/ae.safetensors",
            },
        ],
        "workflow":   "z_image",
        "steps":      8,
        "cfg":        1.0,
        "sampler":    "res_multistep",
        "scheduler":  "simple",
        "short":      "z-image-turbo",
        "tier":       "medium",    # ~20.7 GB complete pipeline
    },
    {
        "label":      "Flux.1-dev",
        "checkpoint": "flux1-dev.safetensors",
        "support_assets": [
            {
                "name": "t5xxl_fp16.safetensors", "folder": "clip",
                "repo": "comfyanonymous/flux_text_encoders",
                "remote": "t5xxl_fp16.safetensors",
            },
            {
                "name": "clip_l.safetensors", "folder": "clip",
                "repo": "comfyanonymous/flux_text_encoders",
                "remote": "clip_l.safetensors",
            },
            {
                "name": "ae.safetensors", "folder": "vae",
                "repo": "black-forest-labs/FLUX.1-schnell",
                "remote": "ae.safetensors", "gated": True,
            },
        ],
        "workflow":   "flux",
        "steps":      20,
        "cfg":        1.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux-dev",
        "tier":       "large",     # ~23.9 GB
        "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
    },
    {
        "label":      "Flux.2-dev",
        "checkpoint": "flux2-dev.safetensors",
        "support_assets": [
            {
                "name": "mistral_3_small_flux2_fp8.safetensors",
                "folder": "text_encoders", "repo": "Comfy-Org/flux2-dev",
                "remote": "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors",
            },
            {
                "name": "flux2-vae.safetensors", "folder": "vae",
                "repo": "Comfy-Org/flux2-dev",
                "remote": "split_files/vae/flux2-vae.safetensors",
            },
        ],
        "workflow":   "flux2",
        "steps":      28,
        "cfg":        4.0,
        "sampler":    "euler",
        "scheduler":  "simple",
        "short":      "flux2-dev",
        "tier":       "large",     # ~64.5 GB
        "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-dev",
    },
]

# Extra-small-tier models (<6B parameters).
LLM_MODELS_XSMALL = sorted([
    {
        "tag":            "gemma3:1b-it-q4_K_M",
        "label":          "Gemma 3 1B",
        "short":          "gemma3-1b",
        "base_model":     "gemma3:1b-it",
        "tier":           "xsmall",
        "download_size":  "~0.8 GB",
        "params_b":       1,
        "hf_repo":        "bartowski/google_gemma-3-1b-it-GGUF",
        "hf_file":        "google_gemma-3-1b-it-Q4_K_M.gguf",
        "vllm_repo":      "gaunernst/gemma-3-1b-it-int4-awq",
        "vllm_download_size": "~1.1 GB",
        "variants": [
            {
                "tag": "gemma3:1b-it-q4_K_M", "short": "gemma3-1b",
                "quantization": "Q4_K_M", "hf_repo": "bartowski/google_gemma-3-1b-it-GGUF",
                "hf_file": "google_gemma-3-1b-it-Q4_K_M.gguf",
                "download_size": "~0.8 GB", "default": True,
            },
            {
                "tag": "gemma3:1b-it-q6_K", "short": "gemma3-1b-q6",
                "quantization": "Q6_K", "hf_repo": "bartowski/google_gemma-3-1b-it-GGUF",
                "hf_file": "google_gemma-3-1b-it-Q6_K.gguf", "download_size": "~1.0 GB",
            },
            {
                "tag": "gemma3:1b-it-q8_0", "short": "gemma3-1b-q8",
                "quantization": "Q8_0", "hf_repo": "bartowski/google_gemma-3-1b-it-GGUF",
                "hf_file": "google_gemma-3-1b-it-Q8_0.gguf", "download_size": "~1.1 GB",
            },
        ],
    },
    {
        "tag":            "granite4.1:3b-q4_K_M",
        "label":          "Granite 4.1 3B",
        "short":          "granite4.1-3b-q4",
        "base_model":     "granite4.1:3b",
        "tier":           "xsmall",
        "download_size":  "~2.1 GB",
        "params_b":       3,
        "hf_repo":        "ibm-granite/granite-4.1-3b-GGUF",
        "hf_file":        "granite-4.1-3b-Q4_K_M.gguf",
        "vllm_repo":      "cyankiwi/granite-4.1-3b-AWQ-INT4",
        "vllm_download_size": "~2.4 GB",
        "vllm_tool_parser": "granite4",
        "variants": [
            {
                "tag": "granite4.1:3b-q4_K_M", "short": "granite4.1-3b-q4",
                "quantization": "Q4_K_M", "hf_repo": "ibm-granite/granite-4.1-3b-GGUF",
                "hf_file": "granite-4.1-3b-Q4_K_M.gguf",
                "download_size": "~2.1 GB", "default": True,
            },
            {
                "tag": "granite4.1:3b-q6_K", "short": "granite4.1-3b-q6",
                "quantization": "Q6_K", "hf_repo": "ibm-granite/granite-4.1-3b-GGUF",
                "hf_file": "granite-4.1-3b-Q6_K.gguf", "download_size": "~2.8 GB",
            },
            {
                "tag": "granite4.1:3b-q8_0", "short": "granite4.1-3b-q8",
                "quantization": "Q8_0", "hf_repo": "ibm-granite/granite-4.1-3b-GGUF",
                "hf_file": "granite-4.1-3b-Q8_0.gguf", "download_size": "~3.7 GB",
            },
        ],
    },
    {
        "tag":            "qwen3.5:4b-q4_K_M",
        "label":          "Qwen3.5 4B",
        "short":          "qwen3.5-4b-q4",
        "base_model":     "qwen3.5:4b",
        "tier":           "xsmall",
        "download_size":  "~3.1 GB",
        "params_b":       4,
        "hf_repo":        "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "hf_file":        "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        "vllm_repo":      "cyankiwi/Qwen3.5-4B-AWQ-4bit",
        "vllm_download_size": "~4.1 GB",
        "native_mtp":     {
            "llamacpp": {"num_speculative_tokens": 3},
            "vllm": {"num_speculative_tokens": 3},
        },
        "variants": [
            {
                "tag": "qwen3.5:4b-q4_K_M", "short": "qwen3.5-4b-q4",
                "quantization": "Q4_K_M", "hf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
                "hf_file": "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
                "download_size": "~3.1 GB", "default": True,
            },
            {
                "tag": "qwen3.5:4b-q6_K", "short": "qwen3.5-4b-q6",
                "quantization": "Q6_K", "hf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
                "hf_file": "Qwen_Qwen3.5-4B-Q6_K.gguf", "download_size": "~3.9 GB",
            },
            {
                "tag": "qwen3.5:4b-q8_0", "short": "qwen3.5-4b-q8",
                "quantization": "Q8_0", "hf_repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
                "hf_file": "Qwen_Qwen3.5-4B-Q8_0.gguf", "download_size": "~4.7 GB",
            },
        ],
    },
], key=lambda m: m["params_b"])

# Small-tier models (≤20B parameters).
# "params_b" is total parameters (not active, for MoE models) and sets sort order below.
LLM_MODELS_SMALL = sorted([
    {
        "tag":            "granite4.1:8b-q4_K_M",
        "label":          "Granite 4.1 8B",
        "short":          "granite4.1-8b-q4",
        "base_model":     "granite4.1:8b",
        "tier":           "small",
        "download_size":  "~5.4 GB",
        "params_b":       8,
        "hf_repo":        "ibm-granite/granite-4.1-8b-GGUF",
        "hf_file":        "granite-4.1-8b-Q4_K_M.gguf",
        "vllm_repo":      "cyankiwi/granite-4.1-8b-AWQ-INT4",
        "vllm_download_size": "~5.5 GB",
        "vllm_tool_parser": "granite4",
        "variants": [
            {
                "tag": "granite4.1:8b-q4_K_M", "short": "granite4.1-8b-q4",
                "quantization": "Q4_K_M", "hf_repo": "ibm-granite/granite-4.1-8b-GGUF",
                "hf_file": "granite-4.1-8b-Q4_K_M.gguf",
                "download_size": "~5.4 GB", "default": True,
            },
            {
                "tag": "granite4.1:8b-q6_K", "short": "granite4.1-8b-q6",
                "quantization": "Q6_K", "hf_repo": "ibm-granite/granite-4.1-8b-GGUF",
                "hf_file": "granite-4.1-8b-Q6_K.gguf", "download_size": "~7.3 GB",
            },
            {
                "tag": "granite4.1:8b-q8_0", "short": "granite4.1-8b-q8",
                "quantization": "Q8_0", "hf_repo": "ibm-granite/granite-4.1-8b-GGUF",
                "hf_file": "granite-4.1-8b-Q8_0.gguf", "download_size": "~9.4 GB",
            },
        ],
    },
    {
        "tag":            "qwen3.5:9b-q4_K_M",
        "label":          "Qwen3.5 9B",
        "short":          "qwen3.5-9b-q4",
        "base_model":     "qwen3.5:9b",
        "tier":           "small",
        "download_size":  "~6.2 GB",
        "params_b":       9,
        "hf_repo":        "bartowski/Qwen_Qwen3.5-9B-GGUF",
        "hf_file":        "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        "vllm_repo":      "cyankiwi/Qwen3.5-9B-AWQ-4bit",
        "vllm_download_size": "~9.1 GB",
        "native_mtp":     {
            "llamacpp": {"num_speculative_tokens": 3},
            "vllm": {"num_speculative_tokens": 3},
        },
        "variants": [
            {
                "tag": "qwen3.5:9b-q4_K_M", "short": "qwen3.5-9b-q4",
                "quantization": "Q4_K_M", "hf_repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
                "hf_file": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
                "download_size": "~6.2 GB", "default": True,
            },
            {
                "tag": "qwen3.5:9b-q6_K", "short": "qwen3.5-9b-q6",
                "quantization": "Q6_K", "hf_repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
                "hf_file": "Qwen_Qwen3.5-9B-Q6_K.gguf", "download_size": "~8.0 GB",
            },
            {
                "tag": "qwen3.5:9b-q8_0", "short": "qwen3.5-9b-q8",
                "quantization": "Q8_0", "hf_repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
                "hf_file": "Qwen_Qwen3.5-9B-Q8_0.gguf", "download_size": "~9.9 GB",
            },
        ],
    },
    {
        "tag":            "gemma4:12b-it-q4_K_M",
        "label":          "Gemma 4 12B",
        "short":          "gemma4-12b-q4",
        "base_model":     "gemma4:12b-it",
        "tier":           "small",
        "download_size":  "~7.7 GB",
        "params_b":       12,
        "hf_repo":        "bartowski/gemma-4-12B-it-GGUF",
        "hf_file":        "gemma-4-12B-it-Q4_K_M.gguf",
        "vllm_repo":      "mattbucci/gemma-4-12B-AWQ",
        "vllm_download_size": "~7.8 GB",
        "vllm_tool_parser": "gemma4",
        "variants": [
            {
                "tag": "gemma4:12b-it-q4_K_M", "short": "gemma4-12b-q4",
                "quantization": "Q4_K_M", "hf_repo": "bartowski/gemma-4-12B-it-GGUF",
                "hf_file": "gemma-4-12B-it-Q4_K_M.gguf",
                "download_size": "~7.7 GB", "default": True,
            },
            {
                "tag": "gemma4:12b-it-q6_K", "short": "gemma4-12b-q6",
                "quantization": "Q6_K", "hf_repo": "bartowski/gemma-4-12B-it-GGUF",
                "hf_file": "gemma-4-12B-it-Q6_K.gguf", "download_size": "~10.3 GB",
            },
            {
                "tag": "gemma4:12b-it-q8_0", "short": "gemma4-12b-q8",
                "quantization": "Q8_0", "hf_repo": "bartowski/gemma-4-12B-it-GGUF",
                "hf_file": "gemma-4-12B-it-Q8_0.gguf", "download_size": "~12.7 GB",
            },
        ],
    },
], key=lambda m: m["params_b"])

# Medium-tier models (26-35B params) — one dense alongside two MoE entries; see docs/workloads.md#dense-vs-mixture-of-experts-moe.
LLM_MODELS_MEDIUM = sorted([
    {
        "tag":            "gemma4:26b-a4b-it-ud-q4_K_M",
        "label":          "Gemma 4 26B-A4B",
        "short":          "gemma4-26b-a4b-q4",
        "base_model":     "gemma4:26b-a4b-it",
        "tier":           "medium",
        "download_size":  "~16.9 GB",
        "params_b":       26,
        "hf_repo":        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "hf_file":        "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "vllm_repo":      "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "vllm_download_size": "~17.2 GB",
        "vllm_tool_parser": "gemma4",
        "variants": [
            {
                "tag": "gemma4:26b-a4b-it-ud-q4_K_M", "short": "gemma4-26b-a4b-q4",
                "quantization": "Q4_K_M", "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "hf_file": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
                "download_size": "~16.9 GB", "default": True,
            },
            {
                "tag": "gemma4:26b-a4b-it-ud-q6_K", "short": "gemma4-26b-a4b-q6",
                "quantization": "Q6_K", "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "hf_file": "gemma-4-26B-A4B-it-UD-Q6_K.gguf", "download_size": "~23.2 GB",
            },
            {
                "tag": "gemma4:26b-a4b-it-q8_0", "short": "gemma4-26b-a4b-q8",
                "quantization": "Q8_0", "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "hf_file": "gemma-4-26B-A4B-it-Q8_0.gguf", "download_size": "~26.9 GB",
            },
        ],
    },
    {
        "tag":            "qwen3.8:27b-ud-q4_K_M",
        "label":          "Qwen 3.8 27B",
        "short":          "qwen3.8-27b-q4",
        "base_model":     "qwen3.8:27b",
        "tier":           "medium",
        "download_size":  "~16.5 GB",
        "params_b":       27,
        "hf_repo":        "unsloth/Qwen3.8-27B-GGUF",
        "hf_file":        "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "vllm_repo":      "pearsonkyle/Qwen3.8-27B-GPTQ-W4A16",
        "vllm_download_size": "~19.5 GB",
        "vllm_tool_parser": "qwen3_xml",
        "native_mtp":     {
            "llamacpp": {
                "num_speculative_tokens": 3,
                "draft_repo": "unsloth/Qwen3.8-27B-GGUF",
                "draft_file": "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
                "draft_download_size": "~1.4 GB",
            },
            "vllm": {
                "method": "qwen3_5_mtp",
                "num_speculative_tokens": 2,
            },
        },
        "variants": [
            {
                "tag": "qwen3.8:27b-ud-q4_K_M", "short": "qwen3.8-27b-q4",
                "quantization": "Q4_K_M", "hf_repo": "unsloth/Qwen3.8-27B-GGUF",
                "hf_file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
                "download_size": "~16.5 GB", "default": True,
            },
            {
                "tag": "qwen3.8:27b-ud-q6_K", "short": "qwen3.8-27b-q6",
                "quantization": "Q6_K", "hf_repo": "unsloth/Qwen3.8-27B-GGUF",
                "hf_file": "Qwen3.8-27B-UD-Q6_K.gguf", "download_size": "~22.0 GB",
            },
            {
                "tag": "qwen3.8:27b-q8_0", "short": "qwen3.8-27b-q8",
                "quantization": "Q8_0", "hf_repo": "unsloth/Qwen3.8-27B-GGUF",
                "hf_file": "Qwen3.8-27B-Q8_0.gguf", "download_size": "~29.1 GB",
            },
        ],
    },
    {
        "tag":            "nemotron3.5-lightning:30b-a3b-ud-q4_K_M",
        "label":          "Nemotron 3.5 Lightning 30B-A3B",
        "short":          "nemotron3.5-lightning-30b-a3b",
        "tier":           "medium",
        "download_size":  "~25.3 GB",
        "params_b":       30,   # 3B active — hybrid Mamba MoE
        "hf_repo":        "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "hf_file":        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf",
        "vllm_repo":      "Local-Axiom-AI/Nemotron-3.5-Lightning-awq",
        "vllm_download_size": "~18.1 GB",
        "native_mtp":     {
            "llamacpp": {"num_speculative_tokens": 1},
            "vllm": {"num_speculative_tokens": 1},
        },
    },
], key=lambda m: m["params_b"])

# Large-tier models (70B+ params), same dense/MoE rationale as medium.
# Qwen3-Coder-Next and Nemotron 3 Super ship as multi-part GGUF splits (see docs/engines.md).
LLM_MODELS_LARGE = sorted([
    {
        "tag":            "llama3.3:70b-instruct-q4_K_M",
        "label":          "Llama 3.3 70B",
        "short":          "llama3.3-70b-q4",
        "base_model":     "llama3.3:70b-instruct",
        "tier":           "large",
        "download_size":  "~39.7 GB",
        "params_b":       70,
        "hf_repo":        "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "hf_file":        "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "vllm_repo":      "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "vllm_download_size": "~39.8 GB",
        "vllm_tool_parser": "llama3_json",
        "variants": [
            {
                "tag": "llama3.3:70b-instruct-q4_K_M", "short": "llama3.3-70b-q4",
                "quantization": "Q4_K_M", "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
                "hf_file": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
                "download_size": "~39.7 GB", "default": True,
            },
            {
                "tag": "llama3.3:70b-instruct-q6_K", "short": "llama3.3-70b-q6",
                "quantization": "Q6_K", "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
                "hf_file": [
                    "Llama-3.3-70B-Instruct-Q6_K/Llama-3.3-70B-Instruct-Q6_K-00001-of-00002.gguf",
                    "Llama-3.3-70B-Instruct-Q6_K/Llama-3.3-70B-Instruct-Q6_K-00002-of-00002.gguf",
                ],
                "download_size": "~57.9 GB",
            },
            {
                "tag": "llama3.3:70b-instruct-q8_0", "short": "llama3.3-70b-q8",
                "quantization": "Q8_0", "hf_repo": "bartowski/Llama-3.3-70B-Instruct-GGUF",
                "hf_file": [
                    "Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf",
                    "Llama-3.3-70B-Instruct-Q8_0/Llama-3.3-70B-Instruct-Q8_0-00002-of-00002.gguf",
                ],
                "download_size": "~75.0 GB",
            },
        ],
    },
    {
        "tag":            "qwen3-coder-next:80b-a3b-q4_K_M",
        "label":          "Qwen3-Coder-Next 80B-A3B",
        "short":          "qwen3-coder-next-80b-a3b-q4",
        "base_model":     "qwen3-coder-next:80b-a3b",
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
        "vllm_repo":      "bullpoint/Qwen3-Coder-Next-AWQ-4bit",
        "vllm_download_size": "~48.3 GB",
        "vllm_tool_parser": "qwen3_coder",
        "variants": [
            {
                "tag": "qwen3-coder-next:80b-a3b-q4_K_M",
                "short": "qwen3-coder-next-80b-a3b-q4", "quantization": "Q4_K_M",
                "hf_repo": "Qwen/Qwen3-Coder-Next-GGUF", "hf_file": [
                    "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
                    "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00002-of-00004.gguf",
                    "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00003-of-00004.gguf",
                    "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00004-of-00004.gguf",
                ], "download_size": "~48.4 GB", "default": True,
            },
            {
                "tag": "qwen3-coder-next:80b-a3b-q6_K",
                "short": "qwen3-coder-next-80b-a3b-q6", "quantization": "Q6_K",
                "hf_repo": "Qwen/Qwen3-Coder-Next-GGUF", "hf_file": [
                    "Qwen3-Coder-Next-Q6_K/Qwen3-Coder-Next-Q6_K-00001-of-00004.gguf",
                    "Qwen3-Coder-Next-Q6_K/Qwen3-Coder-Next-Q6_K-00002-of-00004.gguf",
                    "Qwen3-Coder-Next-Q6_K/Qwen3-Coder-Next-Q6_K-00003-of-00004.gguf",
                    "Qwen3-Coder-Next-Q6_K/Qwen3-Coder-Next-Q6_K-00004-of-00004.gguf",
                ], "download_size": "~65.6 GB",
            },
            {
                "tag": "qwen3-coder-next:80b-a3b-q8_0",
                "short": "qwen3-coder-next-80b-a3b-q8", "quantization": "Q8_0",
                "hf_repo": "Qwen/Qwen3-Coder-Next-GGUF", "hf_file": [
                    "Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00001-of-00004.gguf",
                    "Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00002-of-00004.gguf",
                    "Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00003-of-00004.gguf",
                    "Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-00004-of-00004.gguf",
                ], "download_size": "~84.9 GB",
            },
        ],
    },
    {
        "tag":            "nemotron-3-super:120b",
        "label":          "Nemotron 3 Super 120B",
        "short":          "nemotron3-super-120b",
        "base_model":     "nemotron-3-super:120b-a12b",
        "tier":           "large",
        "download_size":  "~87.0 GB",
        "params_b":       120,   # 12B active — hybrid Mamba-Transformer MoE
        "hf_repo":        "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
        "hf_file":        [
            "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf",
            "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00002-of-00003.gguf",
            "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00003-of-00003.gguf",
        ],
        "vllm_repo":      "cyankiwi/NVIDIA-Nemotron-3-Super-120B-A12B-AWQ-4bit",
        "vllm_download_size": "~80.7 GB",
        "native_mtp":     {"vllm": {"num_speculative_tokens": 1}},
        "variants": [
            {
                "tag": "nemotron-3-super:120b", "short": "nemotron3-super-120b",
                "quantization": "Q4_K_M",
                "hf_repo": "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
                "hf_file": [
                    "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00001-of-00003.gguf",
                    "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00002-of-00003.gguf",
                    "UD-Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_M-00003-of-00003.gguf",
                ], "download_size": "~87.0 GB", "default": True,
            },
            {
                "tag": "nemotron-3-super:120b-ud-q6_K", "short": "nemotron3-super-120b-q6",
                "quantization": "Q6_K",
                "hf_repo": "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
                "hf_file": [
                    "UD-Q6_K/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q6_K-00001-of-00004.gguf",
                    "UD-Q6_K/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q6_K-00002-of-00004.gguf",
                    "UD-Q6_K/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q6_K-00003-of-00004.gguf",
                    "UD-Q6_K/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q6_K-00004-of-00004.gguf",
                ], "download_size": "~114.8 GB",
            },
            {
                "tag": "nemotron-3-super:120b-q8_0", "short": "nemotron3-super-120b-q8",
                "quantization": "Q8_0",
                "hf_repo": "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
                "hf_file": [
                    "Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00001-of-00004.gguf",
                    "Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00002-of-00004.gguf",
                    "Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00003-of-00004.gguf",
                    "Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00004-of-00004.gguf",
                ], "download_size": "~128.5 GB",
            },
        ],
    },
], key=lambda m: m["params_b"])

LLM_MODELS = LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE


def image_checkpoint_folder(model: dict) -> str:
    return model.get("checkpoint_folder", "checkpoints")


def image_checkpoint_path(model: dict, models_dir: Path) -> Path:
    return Path(models_dir) / image_checkpoint_folder(model) / model["checkpoint"]


def image_required_asset_paths(model: dict, models_dir: Path) -> list[Path]:
    paths = [image_checkpoint_path(model, models_dir)]
    paths.extend(
        Path(models_dir) / asset["folder"] / asset["name"]
        for asset in model.get("support_assets", ())
    )
    return paths


def image_checkpoint_loader(model: dict) -> str:
    return model.get("checkpoint_loader", "CheckpointLoaderSimple")


def image_checkpoint_groups(models: list[dict]) -> dict[str, set[str]]:
    groups = {}
    for model in models:
        groups.setdefault(image_checkpoint_loader(model), set()).add(model["checkpoint"])
    return groups


def qualification_llm_model(engine: str) -> dict:
    if engine == "llamacpp":
        return LLM_MODELS[0]
    if engine == "vllm":
        return next(model for model in LLM_MODELS if model.get("vllm_tool_parser"))
    raise ValueError(f"unknown qualification engine: {engine}")
