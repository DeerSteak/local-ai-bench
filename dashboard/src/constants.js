// Colors per known LLM model short key (order follows LLM_MODEL_ORDER, i.e.
// models.py's extra-small -> small -> medium -> large tier order)
export const MODEL_COLORS = {
  "gemma3-1b":              "#00e5ff",
  "granite4.1-3b-q4":       "#18ffff",
  "qwen3.5-4b-q4":          "#b388ff",
  "granite4.1-8b-q4":       "#ffab40",
  "qwen3.5-9b-q4":          "#76ff03",
  "phi4-14b":               "#ff4081",
  "qwen3.6-27b-q4":         "#ff6e40",
  "nemotron3-nano-30b-a3b": "#00e676",
  "qwen3.6-35b-a3b":        "#ea80fc",
  "llama3.3-70b-q4":        "#69f0ae",
  "qwen3-coder-next-80b-a3b-q4": "#ffea00",
  "nemotron3-super-120b":   "#40c4ff",

  // Retained so results created by earlier catalog versions keep stable colors.
  "llama3.2-3b-q4":         "#536dfe",
  "phi4-mini":              "#64ffda",
  "mistral-7b-q4":          "#ff8a80",
  "llama3.1-8b-q4":         "#7c4dff",
  "llama4-16x17b":          "#ffd740",
};

// Matches models.py's IMAGE_MODELS list
export const IMAGE_MODEL_COLORS = {
  "sd15":         "#ff4081",
  "sdxl":         "#00e5ff",
  "sd35-large":   "#7c4dff",
  "flux-dev":     "#ff6d00",
  "flux2-dev":    "#00e676",
};

export const IMAGE_MODEL_LABELS = {
  "sd15":         "Stable Diffusion 1.5",
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

// Categorical palette for single-series bar charts that color each bar by
// row (e.g. accuracy-by-category breakdowns) — same darker/primary family as
// FILE_COLORS/CTX_COLORS rather than the neon FALLBACK_COLORS used for model
// lines, and ordered so hue neighbors in the data (alphabetically adjacent
// category names) don't land next to each other.
export const CATEGORY_COLORS = [
  "#0969da", // blue
  "#e36209", // orange
  "#1a7f37", // green
  "#9a3078", // purple
  "#cf222e", // red
  "#0e7490", // teal
  "#5c6bc0", // indigo
  "#c2185b", // pink
  "#00897b", // teal-green
  "#6a1b9a", // deep purple
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
  "gemma3-1b":              "Gemma 3 1B",
  "granite4.1-3b-q4":       "Granite 4.1 3B Q4_K_M",
  "qwen3.5-4b-q4":          "Qwen3.5 4B Q4_K_M",
  "granite4.1-8b-q4":       "Granite 4.1 8B Q4_K_M",
  "qwen3.5-9b-q4":          "Qwen3.5 9B Q4_K_M",
  "phi4-14b":               "Phi 4 14B",
  "qwen3.6-27b-q4":         "Qwen3.6 27B Q4_K_M",
  "nemotron3-nano-30b-a3b": "Nemotron 3 Nano 30B-A3B",
  "qwen3.6-35b-a3b":        "Qwen3.6 35B-A3B",
  "llama3.3-70b-q4":        "Llama 3.3 70B Q4_K_M",
  "qwen3-coder-next-80b-a3b-q4": "Qwen3-Coder-Next 80B-A3B Q4_K_M",
  "nemotron3-super-120b":   "Nemotron 3 Super 120B",

  // Retained so older results remain human-readable.
  "llama3.2-3b-q4":         "Llama 3.2 3B Q4_K_M",
  "phi4-mini":              "Phi 4 Mini",
  "mistral-7b-q4":          "Mistral 7B v0.3 Q4_K_M",
  "llama3.1-8b-q4":         "Llama 3.1 8B Q4_K_M",
  "llama4-16x17b":          "Llama 4 Scout 16x17B",
};

// Canonical model order (determines chart line order and color assignment).
// Matches models.py: LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM
// + LLM_MODELS_LARGE.
export const LLM_MODEL_ORDER = [
  "gemma3-1b", "granite4.1-3b-q4", "qwen3.5-4b-q4",
  "granite4.1-8b-q4", "qwen3.5-9b-q4", "phi4-14b",
  "qwen3.6-27b-q4", "nemotron3-nano-30b-a3b", "qwen3.6-35b-a3b",
  "llama3.3-70b-q4", "qwen3-coder-next-80b-a3b-q4", "nemotron3-super-120b",
];

export const LEGACY_LLM_MODEL_ORDER = [
  "llama3.2-3b-q4", "phi4-mini", "mistral-7b-q4", "llama3.1-8b-q4",
  "llama4-16x17b",
];

export const LLM_DISPLAY_ORDER = [...LLM_MODEL_ORDER, ...LEGACY_LLM_MODEL_ORDER];

// Size tier per model short key — mirrors models.py's LLM_MODELS_XSMALL /
// _SMALL / _MEDIUM / _LARGE groupings (defined by parameter count; VRAM
// footprint is shown per-model for reference, but tier membership doesn't
// depend on it).
export const MODEL_SIZE_TIER = {
  "gemma3-1b":              "xsmall",
  "granite4.1-3b-q4":       "xsmall",
  "qwen3.5-4b-q4":          "xsmall",
  "granite4.1-8b-q4":       "small",
  "qwen3.5-9b-q4":          "small",
  "phi4-14b":               "small",
  "qwen3.6-27b-q4":         "medium",
  "nemotron3-nano-30b-a3b": "medium",
  "qwen3.6-35b-a3b":        "medium",
  "llama3.3-70b-q4":        "large",
  "qwen3-coder-next-80b-a3b-q4": "large",
  "nemotron3-super-120b":   "large",

  // Retained so older results continue to group into their original tiers.
  "llama3.2-3b-q4":         "xsmall",
  "phi4-mini":              "xsmall",
  "mistral-7b-q4":          "small",
  "llama3.1-8b-q4":         "small",
  "llama4-16x17b":          "large",
};

export const IMAGE_MODEL_ORDER = ["sd15", "sdxl", "sd35-large", "flux-dev", "flux2-dev"];

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
export const SECTIONS = ["llm", "llm_conversation", "concurrency_tool", "concurrency_chat", "accuracy", "embeddings", "images"];
export const SECTION_LABELS = {
  llm: "LLM",
  llm_conversation: "LLM Conversation",
  concurrency_tool: "Concurrency (Tool)",
  concurrency_chat: "Concurrency (Chat)",
  accuracy: "Accuracy",
  embeddings: "Embeddings",
  images: "Images",
};

// Ordered concurrency levels swept by concurrency_benchmark.py, one per
// test — matches scripts/config.py's CONCURRENCY_TOOL_LEVELS/
// CONCURRENCY_CHAT_LEVELS (kept as strings since results JSON keys are
// strings).
export const CONCURRENCY_LEVELS = {
  concurrency_tool: ["1", "2", "4", "6", "8", "12", "16"],
  concurrency_chat: ["1", "2", "4", "8", "16", "24", "32"],
};

// Human-readable reason for ConcurrencyBenchmark.run stopping a model's sweep
// early (results JSON's per-model "stopped_at" field).
export const CONCURRENCY_STOP_LABELS = {
  load_failed: "couldn't load at this level — the model's real ceiling on this hardware",
  crashed:     "the engine crashed during this batch",
  failed:      "the batch failed (e.g. timed out)",
  slow:        "per-request tokens/sec dropped below the slow-model cutoff",
};

// Accuracy sub-tests, grouped under the single "Accuracy" section pill —
// matches benchmark.py's ACCURACY_TESTS / results JSON's top-level
// "mcq"/"math"/"reasoning"/"code"/"tool" keys.
export const ACCURACY_TESTS = ["mcq", "math", "reasoning", "code", "tool"];
export const ACCURACY_TEST_LABELS = {
  mcq: "MCQ", math: "Math", reasoning: "Reasoning", code: "Code", tool: "Tool Use",
};

// Fixed two-series bar config for the accuracy timeout/loop-detection chart
// (scripts/*_benchmark.py's timed_out_count / likely_loop_count fields) —
// not per-model, so this doesn't need a color-assignment helper like the
// other bar configs.
export const ACCURACY_TIMEOUT_BAR_CONFIGS = [
  { dataKey: "timed_out_count",   name: "Timed out",   fill: "#e36209" },
  { dataKey: "likely_loop_count", name: "Likely loop", fill: "#cf222e" },
];

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
  "sd15":         "#cf222e",
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
// only ever uses {0.5K, 2K, 8K, 32K, 64K} (a subsequence of this, in the same
// relative order); LLM Conversation samples up to 96K, including the 48K and
// 80K checkpoints added between the wider steps (128K is kept here only so
// older result files that still have a 128K checkpoint still render).
export const CTX_ORDER = ["0K", "0.5K", "2K", "4K", "8K", "16K", "32K", "48K", "64K", "80K", "96K", "128K"];

// Image resolution order
export const RES_ORDER = ["512x512", "768x768", "1024x1024", "1536x1536"];

// Colors per image resolution (used in "Group by System" bar chart mode)
export const RES_COLORS = {
  "512x512":   "#0969da",
  "768x768":   "#e36209",
  "1024x1024": "#1a7f37",
  "1536x1536": "#9a3078",
};

// Backend badge colors
export const BACKEND_COLORS = {
  cuda:  { bg: "#dafbe1", color: "#3fb950", border: "#aceebb" },
  rocm:  { bg: "#f0ebff", color: "#7c4dff", border: "#c8b8f8" },
  metal: { bg: "#dff0ff", color: "#0969da", border: "#b6d4fb" },
  xpu:   { bg: "#e0f7f7", color: "#0e8a8a", border: "#a8e6e6" },
  vulkan:{ bg: "#fff4e5", color: "#b45309", border: "#f3c98b" },
  cpu:   { bg: "#f6f8fa", color: "#8c959f", border: "#d0d7de" },
};
