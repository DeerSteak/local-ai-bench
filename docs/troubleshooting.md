# Troubleshooting

Start with the exact error and the latest durable result rather than deleting state or rerunning everything. A failed or interrupted run can still contain valid completed cases; preserve the result JSON, event data, and sidecars until their usability has been reviewed.

## Setup and launch

| Symptom | Check | Resolution |
|---|---|---|
| GUI does not open | Whether the session is SSH/headless and whether the active Python has Tk | Use the terminal interface or rerun setup after installing the offered Tk package; GUI absence never blocks CLI use |
| Double-click launcher leaves or closes the wrong terminal | Launcher file and macOS automation permission | Use the shipped launcher from the repository; on first use allow Terminal to control Terminal when macOS asks |
| Setup cannot find llama.cpp tools | Read-only inventory and `PATH`; distinguish server from optional benchmark tools | Supply/install the missing tool explicitly; setup does not replace a working system installation merely because an optional tool is absent |
| Setup cannot find ComfyUI | Saved path, running process, common locations, and whether the path contains `main.py` | Select the actual program directory or portable root; an invalid path falls back to the reviewed managed download |
| Image models are missing in system ComfyUI | `extra_model_paths.yaml` managed block and whether a running server predates it | Stop/restart ComfyUI so it reloads the added managed-model path |
| Download/extraction stops | Free space, proxy/network, archive validation error, upstream availability | Preserve the error, remove only the named partial download if instructed, and retry; never bypass traversal/link validation |

## Benchmark execution

| Symptom | Check | Resolution |
|---|---|---|
| Model appears stuck loading | Progress window model/case, terminal process output, memory pressure, timeout | Allow expected first load; if timed out, cancel normally and inspect preserved prior cases before narrowing model/context scope |
| llama-bench is unexpectedly slow | Model lifecycle, selected matrix, prompt/decode sizes, memory pressure | Confirm the current retained-model behavior and run-plan case count; do not compare with an older methodology without its warning |
| Unrealistic token rate | Retry/invalid sample records | The implausible sample is dropped and retried once; a second implausible sample remains invalid and excluded while the run continues |
| A later native case times out | Earlier streamed case/repetition entries | Earlier successful entries remain checkpointed and usable; coverage and timeout metadata show what is missing |
| Ctrl-C or GUI cancel | Run/stage status and last checkpoint | Wait for cleanup; completed cases remain saved and the active stage becomes interrupted rather than masquerading as complete |
| Result cannot be compared | Plan/methodology/runtime/model identity warning | Compare only compatible identities or explicitly fork/re-run; never suppress an incompatibility to obtain a chart |
| Offline mode blocks work | Whether all models/runtimes were installed before launch | Install prerequisites online through reviewed setup, then rerun offline; offline execution intentionally denies non-loopback Python connections |

## Results, reports, and support

Malformed/non-finite results are rejected instead of partially rewritten. Work from a copy, retain the original, and use the exact validation error. A portable bundle must pass digest, schema, plan, bank, aggregate, path, and size verification before import. Report generation requires explicit outbound metadata review; use system/hardware aliases for embargoed machines and inspect attached artifacts separately.

Use **Export Support Bundle** only after previewing every included file and field. The allowlisted bundle excludes prompts, responses, tokens, credentials, hostnames, model names, and private paths by default, but the user remains responsible for reviewing it against the project's classification and embargo rules. For performance discrepancies, create the vendor diagnostic package from the source result rather than sending an unrestricted terminal log.

When requesting help, include application version, operating system/architecture, disclosed accelerator/backend, selected engine/tests, run status, exact safe error text, whether data was preserved, and the reviewed `.labsupport` bundle if permitted. Never paste `hf.txt`, authorization headers, proprietary prompts/responses, embargoed hardware identity, or unreviewed result/log files into a public issue.

For onboarding without real hardware, `samples/representative_v4_1.labresult` is a deterministic synthetic schema-3 portable bundle with non-private aliases. It exercises bundle verification/import and result inspection only; its measurements are test-fixture data and must never enter a recommendation corpus, hardware claim, acceptance decision, or performance comparison presented as measured evidence.
