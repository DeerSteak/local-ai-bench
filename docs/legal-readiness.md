# Legal and dependency readiness

This document is an engineering inventory and review gate, not legal advice or a declaration that every listed item is cleared for commercial distribution.

## Current distribution facts

The repository's current `LICENSE` is the PolyForm Noncommercial License 1.0.0. That does not grant downstream commercial-use rights. The copyright owner can offer separate commercial terms, but the public license and the intended paid/free boundary must be reconciled before a commercial launch. The Git contribution history currently lists one contributor identity with two email addresses; confirming ownership, employment obligations, and the right to relicense remains a human legal task.

Setup downloads llama.cpp, ComfyUI, model weights, and image checkpoints from their upstream sources instead of treating those artifacts as owned project code. System installations may be reused. Download-on-demand reduces bundled payload but does not remove the need to review each upstream license, model license, acceptable-use term, or redistribution restriction. Until that review is recorded, runtime binaries and models are customer-supplied or separately downloaded and must not be included in a commercial installer.

## Inventory sources

| Class | Authoritative inventory | Current license state |
|---|---|---|
| Owned application code and launchers | Git tracked files and contribution history | PolyForm Noncommercial public license; commercial terms unresolved |
| Python packages | `requirements.txt` and `tests/requirements.txt` | Generated SBOM uses `NOASSERTION` until package metadata is reviewed |
| Dashboard packages and IBM Plex fonts | `dashboard/package-lock.json` | Locked versions, integrity, and declared npm license captured by SBOM; notice review required |
| llama.cpp and ComfyUI | Setup discovery/download logic; vendored `ComfyUI/LICENSE` where present | Upstream license and exact distributed form require review |
| LLM and embedding references | `scripts/workloads/models.py` and the versioned model catalog | License status explicitly unverified; not eligible for supported recommendations |
| Image checkpoints and supporting encoders | `scripts/workloads/models.py` and setup download definitions | Model-specific license review required |
| Accuracy question banks and sample document | `scripts/workloads/data/` and the workload loaders | Provenance/authorship record required before commercial distribution |
| Generated reports, sample results, screenshots, and images | `samples/`, `results/`, and report/export tools | Separate owned, synthetic, customer, and third-party inputs before publication |

## Repeatable SBOM process

Run `python -m scripts.release.sbom local-ai-bench-sbom.json` from the repository root. The deterministic JSON inventories every declared Python requirement and every locked npm transitive package. It preserves versions, npm integrity values, runtime/development scope, declared npm licenses, and unknown Python license facts without inference. Reviewers then resolve every `NOASSERTION`, produce third-party notices from authoritative license texts, and archive both outputs with the release.

Every added or upgraded dependency, executable, dataset, question bank, model, font, or generated third-party asset requires source, exact version/digest, license, redistribution mode, attribution/notice obligations, commercial-use constraints, privacy/security review, and an accountable approver. Unknown or conflicting rights block bundling and supported commercial recommendations; they are not waived by a passing test suite.
