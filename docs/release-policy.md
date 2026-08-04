# Platform support and release policy

Local AI Bench 4.1 is a preview engineering build. A setup or runtime code path is not by itself a commercial support claim. No platform is labeled stable-supported until the qualification evidence below exists for the exact operating-system, architecture, accelerator/backend, runtime, and installer combination.

## Platform matrix

| Platform family | Current status | Candidate qualification configurations | Important bounds |
|---|---|---|---|
| macOS on Apple silicon | Qualification candidate | M4 Pro Mac mini at representative unified-memory sizes; one larger Apple-silicon system | Native arm64, supported macOS release, llama.cpp Metal, ComfyUI MPS |
| Linux on NVIDIA x86-64 | Qualification candidate | Current Ubuntu LTS with a representative consumer RTX GPU | Declared NVIDIA driver/CUDA and llama.cpp build identity |
| Linux on NVIDIA ARM64 | Qualification candidate | DGX Spark | Exact DGX OS/driver stack, GB10, system or managed llama.cpp |
| Linux on AMD Ryzen AI Max | Qualification candidate | Ryzen AI Max+ 395 system | Exact distribution/kernel/ROCm support; system configuration varies by vendor |
| Windows on NVIDIA x86-64 | Qualification candidate | Current Windows 11 with representative RTX GPU | Selected CUDA or disclosed Vulkan fallback; portable ComfyUI identity |
| Windows on AMD x86-64 | Preview | Current Windows 11 with a representative supported Radeon/Ryzen AI system | Official portable ComfyUI support and disclosed llama.cpp backend |
| Linux or Windows on Intel Arc | Experimental | None yet | Current setup limitations in the Setup guide apply; no support commitment |
| CPU-only on listed operating systems | Preview | One representative x86-64 and arm64 system | Functional fallback, not a performance-equivalent accelerator configuration |
| Other operating systems, architectures, accelerators, containers, VMs, and remote desktops | Unsupported | None | May run, but results and lifecycle behavior are not supported |

Each stable-supported matrix row must name exact tested versions and hardware rather than inheriting a family-wide claim. A newer OS, driver, runtime, accelerator generation, or packaging method starts as preview until compatibility evidence is added. Experimental paths can change without stable migration guarantees; unsupported paths receive no qualification promise.

## Qualification record

For each candidate configuration, perform clean install, first launch, setup cancellation and retry, system-runtime reuse, managed-runtime install, default run, custom run, interruption at multiple stages, partial-result recovery, report/bundle creation, offline packet capture, repair, upgrade, rollback, uninstall, and orphan-process inspection. Run representative complete benchmarks independently multiple times and evaluate valid coverage and variability against the declared methodology. Record exact hardware, firmware, OS, drivers, runtime digests, installer digest, model artifacts, environment, failures, support intervention, and evidence links.

The current matrix is the required test design, not completed evidence. Individual rows become stable-supported only through a release decision referencing their completed qualification records.

## Stable-release criteria

A stable release requires all automated Python/dashboard tests; current golden compatibility and methodology review; no unexplained behavioral difference from the qualification baseline; completed supported-row hardware records; signed installers/checksums/provenance; SBOM and reviewed notices; dependency, secret, static, artifact, and vulnerability scans; offline evidence; clean install/repair/upgrade/rollback/uninstall evidence; complete option-coverage and accessibility review; current documentation and migration notes; support/incident contacts; and no unresolved high-severity issue without explicit time-bounded acceptance.

`bench-env/bin/python scripts/release_readiness.py [EVIDENCE.json]` emits a machine-readable preflight and exits nonzero while any required gate remains unresolved. Local checks cover front-end option coverage, model licenses, dependency licenses, and target-hardware qualification. Signed installers, release security scans, offline platform qualification, clean-machine lifecycle, accessibility/usability, legal approval, independent security assessment, and final stable-release approval remain failed unless the optional evidence object supplies `status: "passed"`, a non-empty `approved_by`, `approved_at`, and one or more evidence references for each named gate. The command validates presence and shape, not the truth of an approval; release reviewers must inspect the referenced evidence.

The [release artifact manifest](release-artifacts.md) records deterministic SHA-256 checksums and source provenance for explicit packaged files. Stable verification still requires real protected signing credentials and independent pipeline evidence.

Release approval records the product, methodology, engineering, security, licensing, design, and release-operations decisions separately. A failed gate blocks stable status; it may produce a clearly labeled preview build but cannot be silently waived.

## Release channels and notes

Preview builds are opt-in and may be promoted only after a staged cohort shows acceptable install, valid-run, recovery, and support outcomes. Stable rollout is staged with a halt condition and a tested route back to the last signed stable release. Automatic updates are not enabled until signature verification, downgrade protection, recovery, and rollback are implemented and qualified.

Every release note names these compatibility axes independently: application version; results schema; run-plan schema; event/bundle/report formats; engine adapter contract and runtime versions; workload pack and question-bank versions; methodology profile and baseline; supported/preview/experimental matrix changes; installer/update compatibility; migrations and rollback; known issues; security fixes; and deprecations. A single marketing version must never imply that all schemas or methodologies are interchangeable.
