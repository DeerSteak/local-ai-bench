// Colors per known LLM model short key (order follows LLM_MODEL_ORDER, i.e.
// models.py's extra-small -> small -> medium -> large tier order)
export const MODEL_COLORS = {
  "llama3.2-3b-q4":    "#536dfe",
  "phi4-mini":         "#64ffda",
  "qwen3.5-4b":        "#ff8a80",
  "llama3.1-8b-q4":    "#7c4dff",
  "deepseek-r1-8b":    "#00e5ff",
  "gemma4-e4b":        "#ff6d00",
  "gpt-oss-20b":       "#ff4081",
  "gemma4-26b":        "#00e676",
  "deepseek-r1-32b":   "#ffd740",
  "qwen3.6-35b-a3b":   "#ea80fc",
  "llama3.3-70b-q4":   "#69f0ae",
  "deepseek-r1-70b":   "#ccff90",
  "gpt-oss-120b":      "#40c4ff",
};

// Matches models.py's IMAGE_MODELS list
export const IMAGE_MODEL_COLORS = {
  "sdxl":         "#00e5ff",
  "sd35-large":   "#7c4dff",
  "flux-dev":     "#ff6d00",
  "flux2-dev":    "#00e676",
};

export const IMAGE_MODEL_LABELS = {
  "sdxl":         "SDXL",
  "sd35-large":   "SD3.5 Large",
  "flux-dev":     "Flux.1-dev",
  "flux2-dev":    "Flux.2-dev",
};

// Fallback palette for unknown model keys (hash-assigned)
export const FALLBACK_COLORS = [
  "#00e5ff", "#7c4dff", "#ff6d00", "#00e676",
  "#ff4081", "#ffd740", "#69f0ae", "#40c4ff",
  "#ea80fc", "#ccff90", "#ff6e40", "#80d8ff",
];

// Per-file colors for multi-file comparison (up to MAX_FILES)
export const FILE_COLORS = [
  "#0969da", // blue
  "#e36209", // orange
  "#1a7f37", // green
  "#9a3078", // purple
  "#cf222e", // red
  "#0e7490", // teal
];

// Dash patterns for distinguishing models within a file color
export const MODEL_DASH_PATTERNS = [
  undefined,    // solid
  "8 4",        // long dash
  "3 3",        // dots
  "10 3 3 3",   // dash-dot
  "14 3",       // very long dash
  "3 3 10 3",   // dot-dash
  "6 2",        // short dash
  "10 2 2 2",   // dash-dot-dot
];

export const MAX_FILES = 6;

// Human-readable labels for LLM model short keys (matches the "label" field
// in models.py, the single source of truth for the model roster)
export const LLM_MODEL_LABELS = {
  "llama3.2-3b-q4":  "Llama 3.2 3B Q4_K_M",
  "phi4-mini":       "Phi 4 Mini",
  "qwen3.5-4b":      "Qwen3.5 4B",
  "llama3.1-8b-q4":  "Llama 3.1 8B Q4_K_M",
  "deepseek-r1-8b":  "DeepSeek-R1 8B",
  "gemma4-e4b":      "Gemma 4 E4B",
  "gpt-oss-20b":     "GPT-OSS 20B (MXFP4)",
  "gemma4-26b":      "Gemma 4 26B",
  "deepseek-r1-32b": "DeepSeek-R1 32B",
  "qwen3.6-35b-a3b": "Qwen3.6 35B-A3B",
  "llama3.3-70b-q4": "Llama 3.3 70B Q4_K_M",
  "deepseek-r1-70b": "DeepSeek-R1 70B",
  "gpt-oss-120b":    "GPT-OSS 120B (MXFP4)",
};

// Canonical model order (determines chart line order and color assignment).
// Matches models.py: LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM
// + LLM_MODELS_LARGE.
export const LLM_MODEL_ORDER = [
  "llama3.2-3b-q4", "phi4-mini", "qwen3.5-4b",
  "llama3.1-8b-q4", "deepseek-r1-8b", "gemma4-e4b", "gpt-oss-20b",
  "gemma4-26b", "deepseek-r1-32b", "qwen3.6-35b-a3b",
  "llama3.3-70b-q4", "deepseek-r1-70b", "gpt-oss-120b",
];

// Size tier per model short key — mirrors models.py's LLM_MODELS_XSMALL /
// _SMALL / _MEDIUM / _LARGE groupings (defined by parameter count; VRAM
// footprint is shown per-model for reference, but tier membership doesn't
// depend on it).
export const MODEL_SIZE_TIER = {
  "llama3.2-3b-q4":  "xsmall",
  "phi4-mini":       "xsmall",
  "qwen3.5-4b":      "xsmall",
  "llama3.1-8b-q4":  "small",
  "deepseek-r1-8b":  "small",
  "gemma4-e4b":      "small",
  "gpt-oss-20b":     "small",
  "gemma4-26b":      "medium",
  "deepseek-r1-32b": "medium",
  "qwen3.6-35b-a3b": "medium",
  "llama3.3-70b-q4": "large",
  "deepseek-r1-70b": "large",
  "gpt-oss-120b":    "large",
};

export const IMAGE_MODEL_ORDER = ["sdxl", "sd35-large", "flux-dev", "flux2-dev"];

// Matches models.py's EMBED_MODELS list
export const EMBED_MODEL_COLORS = {
  "nomic-embed-text":  "#0969da",
  "mxbai-embed-large": "#e36209",
};

export const EMBED_MODEL_LABELS = {
  "nomic-embed-text":  "Nomic Embed Text",
  "mxbai-embed-large": "MixedBread Embed Large",
};

export const EMBED_MODEL_ORDER = ["nomic-embed-text", "mxbai-embed-large"];

// Size tiers for splitting up per-system charts (too many models on one line
// chart is unreadable — bar charts don't have this problem since bars stack
// rather than overlapping lines). Matches models.py / README.md's parameter-
// count-based tiers via MODEL_SIZE_TIER above.
export const SIZE_TIER_ORDER = ["xsmall", "small", "medium", "large"];
export const SIZE_TIER_LABELS = {
  xsmall: "Extra-small (<6B params)",
  small:  "Small (≤20B params)",
  medium: "Medium (26–35B params)",
  large:  "Large (70B+ params)",
};

// Sections
export const SECTIONS = ["llm", "llm_conversation", "embeddings", "images"];
export const SECTION_LABELS = {
  llm: "LLM",
  llm_conversation: "LLM Conversation",
  embeddings: "Embeddings",
  images: "Images",
};

// LLM metric options
export const LLM_METRICS = ["tps", "ttft"];
export const LLM_METRIC_LABELS = { tps: "Tokens/sec", ttft: "TTFT (sec)" };

// Colors per context length (used in bar chart mode)
export const CTX_COLORS = {
  "0K":   "#5c6bc0",
  "2K":   "#9a3078",
  "4K":   "#00acc1",
  "8K":   "#0969da",
  "16K":  "#f9a825",
  "32K":  "#e36209",
  "48K":  "#00897b",
  "64K":  "#1a7f37",
  "80K":  "#6a1b9a",
  "96K":  "#8d6e63",
  "128K": "#c2185b",
};

// Colors per image model (used in bar chart mode — matches CTX palette)
export const IMAGE_BAR_COLORS = {
  "sdxl":         "#0969da",
  "sd35-large":   "#e36209",
  "flux-dev":     "#1a7f37",
  "flux2-dev":    "#9a3078",
};

// Colors per embedding model (used in bar chart mode — matches CTX/BATCH palette)
export const EMBED_BAR_COLORS = {
  "nomic-embed-text":  "#0969da",
  "mxbai-embed-large": "#e36209",
};

// Ordered context length labels (match benchmark.py output). The LLM section
// only ever uses {2K, 8K, 32K, 64K} (a subsequence of this, in the same
// relative order); LLM Conversation samples up to 96K, including the 48K and
// 80K checkpoints added between the wider steps (128K is kept here only so
// older result files that still have a 128K checkpoint still render).
export const CTX_ORDER = ["0K", "2K", "4K", "8K", "16K", "32K", "48K", "64K", "80K", "96K", "128K"];

// Image resolution order
export const RES_ORDER = ["1024x1024", "1536x1536"];

// Colors per image resolution (used in "Group by System" bar chart mode)
export const RES_COLORS = {
  "1024x1024": "#0969da",
  "1536x1536": "#e36209",
};

// Backend badge colors
export const BACKEND_COLORS = {
  cuda:  { bg: "#dafbe1", color: "#3fb950", border: "#aceebb" },
  rocm:  { bg: "#f0ebff", color: "#7c4dff", border: "#c8b8f8" },
  metal: { bg: "#dff0ff", color: "#0969da", border: "#b6d4fb" },
  xpu:   { bg: "#e0f7f7", color: "#0e8a8a", border: "#a8e6e6" },
  cpu:   { bg: "#f6f8fa", color: "#8c959f", border: "#d0d7de" },
};
