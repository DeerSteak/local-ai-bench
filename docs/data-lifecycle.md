[← Back to README](../README.md)

# Local data lifecycle

Local AI Bench stores benchmark state locally by default and does not upload results. The user chooses if and where an artifact leaves the machine.

| Data | Default location | Retention and deletion | Portability |
|---|---|---|---|
| Setup configuration | `.local_ai_bench_setup.json` | Retained until replaced or deleted by the user | Machine-local; not included in portable presets |
| Benchmark frontend state | `.benchmark_frontend_state.json` | Retained until replaced or deleted by the user | Machine-local; paths excluded from portable presets |
| Results | `results/results_*.json` or `--out` | Retained until the user deletes it; interruption keeps the same checkpointed file | Export/import through verified `.labresult` bundles |
| Workload event journal | Beside the result as `*.events.sqlite3` | Retained with the result; deleting the run from Result History removes it with the JSON | Local transactional source for every workload, including accuracy answers, embedding timings without vectors, image artifact references, and native rows; portable JSON, answer sidecars, and bundles remain the interchange format |
| Private execution context | Beside the journal as `*.events.sqlite3.local.json` | Retained only for local recovery; deleting the run removes it | Owner-only local ComfyUI and generated-image paths bound to one job; rejected from result bundles and absent from portable JSON/reports |
| Accuracy answer sidecars | Beside the selected result | Retained with the result; deleting the run from Result History removes its exact sidecars | Optional content-addressed bundle artifacts |
| Generated images | Result-specific image directory | Retained with the result; deleting the run from Result History removes the directory | Optional content-addressed bundle artifacts |
| Crash caches | Gitignored `.*_crash_cache.json` repository files | Retained to avoid repeat crashes; **Clear Crash Caches** on the Configuration tab confirms and deletes every workload type together | Not exported by default |
| Portable presets | User-selected path | Retained until user deletion | Versioned JSON without credentials or private paths |
| Support bundles | User-selected `.labsupport` path | Created only after field/file review; retained until user deletion | Redacted diagnostics only; not a result substitute |

Deleting a highlighted run from the GUI's **Result History** asks for confirmation and removes the selected JSON, its event journal, accuracy-answer sidecars, generated-image directory, and locally derived regraded copies. The deletion uses exact names derived from the selected result rather than wildcards, and a failure retains the main JSON when possible so the operation remains visible and retryable. Separately exported bundles and reports, global crash caches, and ComfyUI's own external output directory are not run-owned history artifacts and are not deleted automatically. The product performs no hidden cloud retention and has no remote-delete promise because it has no hosted sync. Secure deletion characteristics depend on the filesystem and storage device; ordinary deletion is not forensic erasure.

The transactional migration includes a local content-addressed object store for large workload artifacts. Image generation activates it for each successfully saved representative PNG under the run-owned image directory; object references disclose only digest, size, and media type, while source names and paths are not retained in the journal. Other large logs, responses, and exports remain outside the store until their owning workload is migrated explicitly.

Results can contain hardware identity, model names, scores, generated content, and paths chosen by the user and may be confidential or embargoed. Result-bundle and report workflows require an explicit identity-metadata review and support private system/hardware aliases while retaining a verifiable source digest. This does not redact measurements or arbitrary artifacts. Use the separately allowlisted support-bundle workflow for diagnostics; it excludes raw measurements and content by design.
