// Colors per known LLM model short key
export const MODEL_COLORS = {
  "llama3.1-8b-q3":  "#00e5ff",
  "llama3.1-8b-q4":  "#7c4dff",
  "qwen3-14b-q4":    "#ff6d00",
  "qwen3-14b-q8":    "#00e676",
  "gpt-oss-20b":     "#ff4081",
  "llama3.1-70b-q3": "#ffd740",
  "llama3.1-70b-q4": "#69f0ae",
  "gpt-oss-120b":    "#40c4ff",
};

export const IMAGE_MODEL_COLORS = {
  "sdxl":         "#00e5ff",
  "flux-schnell": "#7c4dff",
  "flux-dev":     "#ff6d00",
};

export const IMAGE_MODEL_LABELS = {
  "sdxl":         "SDXL",
  "flux-schnell": "Flux.1-schnell",
  "flux-dev":     "Flux.1-dev",
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

// Human-readable labels for LLM model short keys
export const LLM_MODEL_LABELS = {
  "llama3.1-8b-q3":  "Llama 3.1 8B Q3_K_M",
  "llama3.1-8b-q4":  "Llama 3.1 8B Q4_K_M",
  "qwen3-14b-q4":    "Qwen3 14B Q4_K_M",
  "qwen3-14b-q8":    "Qwen3 14B Q8_0",
  "gpt-oss-20b":     "GPT-OSS 20B",
  "llama3.1-70b-q3": "Llama 3.1 70B Q3_K_M",
  "llama3.1-70b-q4": "Llama 3.1 70B Q4_K_M",
  "gpt-oss-120b":    "GPT-OSS 120B",
};

// Canonical model order (determines chart line order and color assignment)
export const LLM_MODEL_ORDER = [
  "llama3.1-8b-q3", "llama3.1-8b-q4",
  "qwen3-14b-q4",   "qwen3-14b-q8",
  "gpt-oss-20b",
  "llama3.1-70b-q3","llama3.1-70b-q4",
  "gpt-oss-120b",
];

export const IMAGE_MODEL_ORDER = ["sdxl", "flux-schnell", "flux-dev"];

// Sections
export const SECTIONS = ["llm", "embeddings", "images"];
export const SECTION_LABELS = { llm: "LLM", embeddings: "Embeddings", images: "Images" };

// LLM metric options
export const LLM_METRICS = ["tps", "ttft"];
export const LLM_METRIC_LABELS = { tps: "Tokens/sec", ttft: "TTFT (sec)" };

// Ordered context length labels (match benchmark.py output)
export const CTX_ORDER = ["8K", "32K", "64K"];

// Embeddings batch keys and labels
export const EMBED_BATCH_KEYS = ["batch_32", "batch_128", "batch_512"];
export const EMBED_BATCH_LABELS = { batch_32: "32", batch_128: "128", batch_512: "512" };

// Image resolution order
export const RES_ORDER = ["1024x1024", "1536x1536"];

// Backend badge colors
export const BACKEND_COLORS = {
  cuda:  { bg: "#dafbe1", color: "#3fb950", border: "#aceebb" },
  rocm:  { bg: "#f0ebff", color: "#7c4dff", border: "#c8b8f8" },
  metal: { bg: "#dff0ff", color: "#0969da", border: "#b6d4fb" },
  cpu:   { bg: "#f6f8fa", color: "#8c959f", border: "#d0d7de" },
};
