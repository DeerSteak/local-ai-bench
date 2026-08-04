[← Back to README](../README.md)

# Local data lifecycle

Local AI Bench stores benchmark state locally by default and does not upload results. The user chooses if and where an artifact leaves the machine.

| Data | Default location | Retention and deletion | Portability |
|---|---|---|---|
| Setup configuration | `.local_ai_bench_setup.json` | Retained until replaced or deleted by the user | Machine-local; not included in portable presets |
| Benchmark frontend state | `.benchmark_frontend_state.json` | Retained until replaced or deleted by the user | Machine-local; paths excluded from portable presets |
| Results | `results/results_*.json` or `--out` | Retained until the user deletes it; interruption keeps the same checkpointed file | Export/import through verified `.labresult` bundles |
| Accuracy answer sidecars | Beside the selected result | Retained and deleted independently with the result set | Optional content-addressed bundle artifacts |
| Generated images | Result-specific image directory | Retained and deleted independently with the result set | Optional content-addressed bundle artifacts |
| Crash caches | Gitignored repository files | Retained to avoid repeat crashes; safe for the user to delete to force a retry | Not exported by default |
| Portable presets | User-selected path | Retained until user deletion | Versioned JSON without credentials or private paths |
| Support bundles | User-selected `.labsupport` path | Created only after field/file review; retained until user deletion | Redacted diagnostics only; not a result substitute |

Deleting a result does not automatically delete separately selected bundles, reports, sidecars, images, or crash caches. The current product performs no hidden cloud retention and has no remote-delete promise because it has no hosted sync. Secure deletion characteristics depend on the filesystem and storage device; ordinary deletion should not be described as forensic erasure.

The transactional migration includes a local content-addressed object store for future large logs, responses, images, and exports. Object references disclose only digest, size, and media type; source names and paths are not retained. It remains inactive for current workloads until each workload moves from JSON ownership, so current files are neither duplicated nor silently retained there.

Results can contain hardware identity, model names, scores, generated content, and paths chosen by the user and may be confidential or embargoed. Review ordinary JSON and result bundles before transfer. Use the separately allowlisted support-bundle workflow for diagnostics; it excludes raw measurements and content by design.
