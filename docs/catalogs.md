# Product catalogs

Local AI Bench keeps a versioned product catalog in `scripts/results/catalogs.py`. Stable hardware and model IDs let future recommendation evidence refer to a specific product or artifact without depending on a display label. Catalog version `1` introduced the three primary small-system targets: Apple Mac mini with M4 Pro, NVIDIA DGX Spark, and the AMD Ryzen AI Max+ 395 processor platform. Catalog version `2` refreshed the medium LLM tier while retaining the same hardware entries. Catalog version `3` expands every llama.cpp quantization variant into its own artifact record so license review and recommendation gates cover all 36 downloadable LLM artifacts.

Hardware specifications come from the linked manufacturer product pages. An entry describes only the configuration facts stated there; for example, the Ryzen processor entry deliberately does not claim a system memory capacity because that depends on the system vendor's configuration. Every initial hardware entry is `unqualified` until Local AI Bench has representative test evidence for it. Current prices are not stored because they vary by seller, region, and date.

The model catalog is generated from `scripts/workloads/models.py`, the existing source of benchmark artifacts. Each record includes a stable artifact identity, known quantization, runtime and workload compatibility, and explicit placeholders for context, memory, and license evidence. Context and memory are detected or estimated at runtime rather than asserted as fixed catalog facts.

## llama.cpp quantization variants

A llama.cpp catalog record may optionally declare a stable `base_model` and two or more `variants`. Every variant repeats its unique tag, short key, quantization label, repository, GGUF filename or complete split-file list, and rounded download size; exactly one is the default and must match the record's legacy top-level acquisition fields. All twelve base models expose Q4_K_M, a preferred Q6-family file, and Q8_0 from their existing GGUF repository, with Q4_K_M retaining its ordinary-run identity and default. The Q6 preference order is Q6_K_XL, then Q6_K_M, then standard Q6_K; Q6_K_L is not selected by this policy.

| Base-model selector | Default | Preferred Q6 family | Q8 | Approximate downloads |
|---|---|---|---|---|
| `gemma3:1b-it` | Q4_K_M | Q6_K | Q8_0 | 0.8 / 1.0 / 1.1 GB |
| `granite4.1:3b` | Q4_K_M | Q6_K | Q8_0 | 2.1 / 2.8 / 3.7 GB |
| `qwen3.5:4b` | Q4_K_M | Q6_K | Q8_0 | 3.1 / 3.9 / 4.7 GB |
| `granite4.1:8b` | Q4_K_M | Q6_K | Q8_0 | 5.4 / 7.3 / 9.4 GB |
| `qwen3.5:9b` | Q4_K_M | Q6_K | Q8_0 | 6.2 / 8.0 / 9.9 GB |
| `gemma4:12b-it` | Q4_K_M | Q6_K | Q8_0 | 7.7 / 10.3 / 12.7 GB |
| `gemma4:26b-a4b-it` | Q4_K_M | Q6_K_XL | Q8_0 | 16.9 / 23.3 / 26.9 GB |
| `qwen3.8:27b` | Q4_K_M | Q6_K_XL | Q8_0 | 16.5 / 25.3 / 29.1 GB |
| `nemotron3.5-lightning:30b-a3b` | Q4_K_M | Q6_K_XL | Q8_0 | 25.3 / 35.1 / 35.1 GB |
| `llama3.3:70b-instruct` | Q4_K_M | Q6_K | Q8_0 | 39.7 / 57.9 / 75.0 GB |
| `qwen3-coder-next:80b-a3b` | Q4_K_M | Q6_K | Q8_0 | 48.4 / 65.6 / 84.9 GB |
| `nemotron-3-super:120b-a12b` | Q4_K_M | Q6_K_XL | Q8_0 | 87.0 / 117.9 / 128.5 GB |

Sizes are rounded download estimates in default/Q6/Q8 order, not runtime-memory promises. The repository and exact filename or split-file set remain authoritative in `scripts/workloads/models.py`; the table documents the user-facing selection contract without duplicating mutable download URLs.

The Version 6 medium tier replaces Gemma 3 27B, Qwen 3.6 35B-A3B, and Nemotron Cascade 2 with Gemma 4 26B-A4B, Qwen 3.8 27B, and Nemotron 3.5 Lightning 30B-A3B. The accepted llama.cpp artifacts completed the catalog compatibility screen on GeForce RTX 5090 under WSL2; hardware-specific capacity or performance limits do not change the catalog identity. Muse Glimmer remains available through custom import rather than expanding the default lineup. Retired IDs remain in the dashboard's legacy registry so historical result files continue to render.

Model artifacts are downloaded directly from their source repositories at the user's request and are not distributed with Local AI Bench. Their catalog records therefore use `download_by_reference`, and an unidentified model license does not block release readiness or supported recommendations. Gated repositories still require the user to review and accept their upstream terms before setup can download them. Any future artifact bundled with the application must have a verified license record before release.

The initial manufacturer sources are [Apple Mac mini specifications](https://www.apple.com/mac-mini/specs/), [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), and [AMD Ryzen laptop processor specifications](https://www.amd.com/en/products/processors/laptop/ryzen.html).
