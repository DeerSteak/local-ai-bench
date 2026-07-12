# Local AI Bench v1.5

Cross-platform benchmarking for LLM generation, image generation, and embeddings. Designed to run on any hardware from an 8GB GPU up to high-memory unified-memory systems — models that don't fit are skipped automatically with no configuration needed.

---

## Quick Start

```bash
git clone https://github.com/DeerSteak/local-ai-bench
cd local-ai-bench
```

| Platform | Script | What it can install |
|---|---|---|
| macOS | `bash setup.sh` | Homebrew, Python |
| Linux / DGX Spark | `bash setup.sh` | Python, Ollama |
| Windows | `setup.bat` | Python, Ollama, ComfyUI portable |

`setup.sh` / `setup.bat` show exactly what they need to install and ask before doing it — nothing happens silently. They then hand off to an interactive model picker, so you choose what gets downloaded before anything installs unattended.

Once setup finishes:

```bash
# Linux / macOS
bash run_linux_mac.sh

# Windows
run_windows.bat
```

A full run takes several hours, depending on your hardware and which options you select. When it's done, explore the results in the [dashboard](docs/dashboard.md):

```bash
python launch_dashboard.py
```

For platform-specific notes, the HuggingFace token flow, and what setup actually installs, see [Setup](docs/setup.md).

---

## Documentation

| Doc | Covers |
|---|---|
| [Setup](docs/setup.md) | What the setup scripts install, the model picker, HuggingFace tokens, platform-specific notes |
| [Workloads](docs/workloads.md) | What's tested — LLM tiers and test modes, image models, embedding models |
| [CLI Reference](docs/cli-reference.md) | Every flag, with examples |
| [Dashboard](docs/dashboard.md) | Loading results, chart sections, what each chart means, exporting |
| [How It Works](docs/how-it-works.md) | Execution order, code organization, full parameter table |
| [Project Structure](docs/project-structure.md) | What every file and folder in the repo is for |

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for non-commercial use, forking, and modification. Commercial licensing: [beatclikr@gmail.com](mailto:beatclikr@gmail.com).
