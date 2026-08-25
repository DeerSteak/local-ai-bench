# Platform support and release policy

Release source moves from `develop` through a versioned release branch into the stable `main` branch. See the [Contributor Workflow](contributor-workflow.md) for branch roles, pull requests, merge direction, tagging, hotfix handling, and repository protection.

Local AI Bench 6.0 is a preview engineering build. A setup or runtime code path is not by itself a commercial support claim. No platform is labeled stable-supported until the qualification evidence below exists for the exact operating-system, architecture, accelerator/backend, runtime, and installer combination.

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

The generated matrices in [Engines](engines.md#qualification-matrix) and [Setup](setup.md#qualification-matrix) are the published authority. `scripts/release/qualification.py` stores exact qualification evidence and derives supported, experimental, or unverified status; the generated sections must never be edited by hand. An entry is supported only when its ordinary benchmark result contains complete measured evidence for every required smallest-model workload and its suite version is no more than one minor release old. Clean install, repair, upgrade, rollback, uninstall, packaging, and offline readiness remain separate release gates. Stale evidence is experimental, and absent or incomplete evidence is unverified.

The [qualification launcher](qualification.md) runs the shipped setup and benchmark paths and requires populated evidence from every compatible workload using the smallest model that supports the engine's complete required workload set. This proves breadth of functional operation without running the full catalog; model-specific compatibility, comparative performance, full-bank accuracy, and production-duration soak claims require their own broader evidence.

## Stable-release criteria

A stable release requires all automated Python/dashboard tests; current golden compatibility and methodology review; no unexplained behavioral difference from the qualification baseline; completed supported-row hardware records; signed installers/checksums/provenance; SBOM and reviewed notices; dependency, secret, static, artifact, and vulnerability scans; offline evidence; clean install/repair/upgrade/rollback/uninstall evidence; complete option-coverage and accessibility review; current documentation and migration notes; support/incident contacts; and no unresolved high-severity issue without explicit time-bounded acceptance.

`python -m scripts.release.release_readiness [EVIDENCE.json]` emits a machine-readable preflight and exits nonzero while any required gate remains unresolved. Local checks cover front-end option coverage, model licenses, dependency licenses, target-hardware qualification, and exact agreement between the generated support matrix and qualification evidence; a manually introduced supported claim therefore fails readiness. Signed installers, release security scans, offline platform qualification, clean-machine lifecycle, accessibility/usability, legal approval, independent security assessment, and final stable-release approval remain failed unless the optional evidence object supplies `status: "passed"`, a non-empty `approved_by`, `approved_at`, and one or more evidence references for each named gate. A separate `telemetry_qualification` object must contain `memory`, `power`, `temperature`, and `combined` records using `paired_observer_v1`, at least 20 trial pairs, a positive interval, non-empty source and platform-class lists, approval metadata, and evidence references. The command validates presence and shape, not the truth of an approval; release reviewers must inspect the referenced evidence.

The [release artifact manifest](release-artifacts.md) records deterministic SHA-256 checksums and source provenance for explicit packaged files. Stable verification still requires real protected signing credentials and independent pipeline evidence.

Release approval records the product, methodology, engineering, security, licensing, design, and release-operations decisions separately. A failed gate blocks stable status; it may produce a clearly labeled preview build but cannot be silently waived.

## Version sync hook

`VERSION` in [`scripts/runtime/config.py`](../scripts/runtime/config.py) is the single source of truth for the application version. Every other place the version appears is a mirror generated from it: the `# Local AI Bench vX.Y` title in `README.md`, and the dashboard's `SUITE_VERSION`, which `dashboard/vite.config.js` already parses out of `config.py` at build time.

`.githooks/pre-commit` runs `python -m scripts.release.version_sync` on every commit. It rewrites any mirror that disagrees with `VERSION` and stages the rewrite, so bumping `VERSION` alone is enough to release a new version. It refuses the commit — printing which file and which string — when the version was edited in a mirror instead of in `config.py`, so a bump can never enter the repo through the wrong file. Mirror drift that predates the commit is repaired rather than blocked.

Hooks are not cloned with a repository, so enable the hook once per working copy:

```bash
git config core.hooksPath .githooks
```

Mirrors include the README title and the doc sentences that state the application version in prose (`docs/telemetry.md`, `docs/security-and-privacy.md`, `docs/maintenance.md`, `docs/release-policy.md`, `docs/product-requirements.md`). Each doc mirror is registered as an explicitly anchored prefix/suffix pair rather than a bare version match, because the docs also carry frozen non-application versions — the 4.1 methodology baseline, the result/run-plan schema axes, `result-compatibility-v4.1.md`, workload-pack and API contract versions — which must never be rewritten by a release bump. Registering a mirror is therefore a deliberate act: a new doc sentence that names the application version is not synced until it is added.

To register a new mirror, add a `VersionTarget` to `TARGETS` in [`scripts/release/version_sync.py`](../scripts/release/version_sync.py) with a regex capturing prefix, version, and trailing text as groups 1–3, or use the `_prose(path, prefix, suffix, description)` helper (a leading `^` in the prefix anchors it to the start of a line). Several targets may share one file.

## Pyright hook

The same `.githooks/pre-commit` also type-checks the project with [pyright](https://microsoft.github.io/pyright/) whenever the commit stages at least one `.py` file, and refuses the commit if it reports any error. It prefers a `pyright` already on `PATH`; otherwise it runs the pinned fallback `npx --yes pyright@1.1.411`, so nothing needs installing to use it without introducing a mutable tool version. `pyrightconfig.json` at the repo root (`typeCheckingMode: "standard"`, `scripts/`+`tests/`) is the config both this hook and an editor's Pyright/Pylance extension pick up automatically. A commit with no staged Python files skips the check entirely, so doc-only or dashboard-only commits stay fast. `git commit --no-verify` bypasses both this and the version-sync check above — reserve it for a confirmed pyright false positive, not for an inconvenient real error.

## Dashboard tsc hook

The same `.githooks/pre-commit` also type-checks the dashboard with `tsc --noEmit` whenever the commit stages at least one `dashboard/src/*.ts`/`*.tsx` file, and refuses the commit if it reports any error. It prefers `dashboard/node_modules/.bin/tsc`; otherwise it runs the pinned fallback `npx --yes typescript@6.0.3 tsc`. `dashboard/tsconfig.json` (`strict: true`, `src/`) is the config both this hook and an editor's TypeScript extension pick up automatically. A commit with no staged dashboard TS files skips the check entirely, so Python-only or doc-only commits stay fast. `git commit --no-verify` bypasses this and every other hook below — reserve it for a confirmed false positive, not for an inconvenient real error.

## `any` ratchet hook

`any` still legitimately appears at the dashboard's results-JSON boundary (see AGENTS.md's TypeScript section) rather than a strict schema, so `.githooks/pre-commit` runs `dashboard/check_any_ratchet.mjs` right after the tsc check above, same trigger (a staged `dashboard/src/*.ts`/`*.tsx` file): a brand-new file must not contain `any` at all, and an existing file's `any` count (a crude but effective `\bany\b` count, comments stripped) may only stay the same or go down, never up. A commit with no staged dashboard TS files skips the check. `git commit --no-verify` bypasses it same as the other hooks above.

## Release channels and notes

Preview builds are opt-in and may be promoted only after a staged cohort shows acceptable install, valid-run, recovery, and support outcomes. Stable rollout is staged with a halt condition and a tested route back to the last signed stable release. Automatic updates are not enabled until signature verification, downgrade protection, recovery, and rollback are implemented and qualified.

Every release note names these compatibility axes independently: application version; results schema; run-plan schema; event/bundle/report formats; engine adapter contract and runtime versions; workload pack and question-bank versions; methodology profile and baseline; supported/preview/experimental matrix changes; installer/update compatibility; migrations and rollback; known issues; security fixes; and deprecations. A single marketing version must never imply that all schemas or methodologies are interchangeable.
