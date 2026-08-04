# Product catalogs

Local AI Bench keeps a versioned product catalog in `scripts/catalogs.py`. Stable hardware and model IDs let future recommendation evidence refer to a specific product or artifact without depending on a display label. Catalog version `1` starts with the three primary small-system targets: Apple Mac mini with M4 Pro, NVIDIA DGX Spark, and the AMD Ryzen AI Max+ 395 processor platform.

Hardware specifications come from the linked manufacturer product pages. An entry describes only the configuration facts stated there; for example, the Ryzen processor entry deliberately does not claim a system memory capacity because that depends on the system vendor's configuration. Every initial hardware entry is `unqualified` until Local AI Bench has representative test evidence for it. Current prices are not stored because they vary by seller, region, and date.

The model catalog is generated from `scripts/models.py`, the existing source of benchmark artifacts. Each record includes a stable artifact identity, known quantization, runtime and workload compatibility, and explicit placeholders for context, memory, and license evidence. Context and memory are detected or estimated at runtime rather than asserted as fixed catalog facts.

An unverified license is a hard eligibility block for supported recommendations. Adding a model to the benchmark catalog does not establish that its weights, source model, quantization, or generated output may be used commercially; license identifiers and authoritative source links must be reviewed and recorded before a model can drive a supported recommendation.

The initial manufacturer sources are [Apple Mac mini specifications](https://www.apple.com/mac-mini/specs/), [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), and [AMD Ryzen laptop processor specifications](https://www.amd.com/en/products/processors/laptop/ryzen.html).
