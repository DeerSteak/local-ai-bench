[← Back to README](../README.md)

# Platform qualification automation

Platform qualification uses a reviewed JSON recipe and a resumable runner. The runner automates lifecycle and representative functional evidence; it does not turn smallest-model coverage into a full-catalog performance claim.

## Scope

A lifecycle recipe runs the smallest compatible model through every workload supported by the selected engine. Shared coverage includes single-shot and conversational LLM generation, embeddings, all five accuracy banks, both server concurrency shapes, and the shortened 120-second sustained path; llama.cpp additionally covers both native benchmark paths and image generation through Stable Diffusion 1.5, while vLLM covers its native `vllm bench` path. Accuracy uses one deterministic question per bank, repeated workloads use one measured run without a warmup, and prompt sweeps stop at 2K. The run verifies that every required result section is populated before it can pass; this is broad functional coverage, not representative performance or full-bank accuracy evidence.

Use a separate performance qualification when a claim depends on catalog-wide compatibility, comparative throughput, full-bank accuracy, production-duration sustained behavior, or model-specific behavior.

## Launchers

Use a disposable clone and run `./run_qualification.sh` on macOS, Linux, or WSL2, or `run_qualification.bat` on Windows. The launcher detects the machine, selects every applicable runtime, uses the repository's reviewed version pins, previews host prerequisites, creates the dedicated `qualification-env`, installs `requirements.txt`, generates concrete recipes, and prints the complete plan. It refuses an unknown accelerator identity, which keeps Radeon and Intel Vulkan evidence separate. WSL2 distinguishes NVIDIA CUDA from AMD ROCm and currently recognizes the Radeon RX 9060 XT as its reviewed RDNA4 qualification target.

Review the preview, then run the same launcher with `--execute`. Execution installs host prerequisites, the selected runtime builds, the smallest LLM and embedding models, and the smallest image model plus an isolated ComfyUI runtime where applicable, then completes the entire resumable lifecycle unattended:

```bash
./run_qualification.sh
./run_qualification.sh --execute
./run_qualification.sh --execute --vllm-only
```

On Windows:

```text
run_qualification.bat
run_qualification.bat --execute
```

The main launcher calls `bootstrap_qualification.sh` or `bootstrap_qualification.bat` itself. The bootstrap covers Git, Python, venv support, CMake, a C++ compiler, and the OpenMPI runtime required by the reviewed ROCm PyTorch wheel where applicable. It deliberately does not alter GPU drivers, CUDA/ROCm SDKs, firmware, or reboot state; those platform-image prerequisites require administrator review because silently replacing them would invalidate the identity being qualified.

The reviewed pins are llama.cpp `b10486 → b10488`, CUDA vLLM `0.27.0 → 0.27.1`, ROCm vLLM `0.27.1+rocm723`, and DGX Spark CUDA 13 nightly `0.26.1rc1.dev950+gcba06764d`. DGX qualification resolves its nightly through the immutable full-commit index rather than the moving `nightly` alias. Wheel channels that retain only one suitable current artifact reinstall that exact build for the update/rollback mechanics instead of selecting an older or incompatible package. Updating these pins is a reviewed code change; qualification never floats to an unreviewed release at execution time.

## Recipe internals

The launchers generate the production recipe; [`samples/qualification_recipe_example.json`](../samples/qualification_recipe_example.json) documents its schema for manual integrations. Commands are JSON arrays and are executed directly without a shell. Give install, upgrade, and rollback commands an isolated qualification prefix; never point them at the normal installation or model store. Legacy recipes containing an uninstall step remain readable, but the runner does not execute it. The recipe may record only the allowlisted non-secret runtime environment fields; credentials such as a Hugging Face token must be inherited at execution time and never written into the recipe or evidence.

The bundled install step uses `scripts.release.qualification_install` to install the selected llama.cpp or vLLM runtime and download the recorded minimum coverage models beneath a disposable repository clone. llama.cpp targets also receive an isolated `qualification-comfyui-runtime`; vLLM targets omit that engine-independent image path because the same platform's llama.cpp row owns it. Runtime installation still obeys the platform support checks; an unsupported vLLM combination fails instead of falling back to another backend. A newly installed vLLM environment must complete a cold import before model downloads begin, so a missing native dependency fails early with its loader detail instead of wasting the model-transfer time. Every engine installation requires `--runtime-version`; vLLM records the complete wheel identity, including a ROCm local-version suffix such as `0.27.1+rocm723`, while llama.cpp records its exact `bNNNNN` release. Qualification never installs a floating latest build. A vLLM recipe sets `HF_HOME` to the disposable clone's `qualification-vllm-cache` directory so installation and the later functional run resolve the same weights.

On a fresh machine, Python 3.11 or newer and Git are bootstrap prerequisites because the qualification code cannot run before the repository and interpreter exist. The POSIX launcher accepts the newest supported interpreter present, including Python 3.14, for `qualification-env`; when a Linux or WSL2 host lacks Python 3.12, executable bootstrap installs a private uv-managed 3.12 for the isolated vLLM environment because the pinned CUDA wheel does not support 3.14 and the pinned ROCm/DGX builds specifically require 3.12. This does not replace the system interpreter or install a different GPU stack. The launcher refreshes `requirements.txt` on every invocation and then installs the runtime and minimum coverage models. Host package-manager changes require an explicit platform-administrator action; credentials and interactive operating-system permissions are never bypassed by the qualification runner.

Cold vLLM environment probes allow five minutes because the first import may initialize large CUDA or ROCm libraries, especially from a Windows-mounted WSL2 clone. This timeout applies only to environment validation, not benchmark requests.

The cancellation command is the only command that may define `interrupt_when_log_contains`. The runner launches it in its own process group, waits for the structured model-running progress event in its live log, sends the platform interrupt signal, and accepts only the declared exit codes. Slow installation, model loading, or server startup therefore cannot cause a premature cancellation.

## Preview and execution

The low-level runner remains preview-first. The top-level launcher writes only the generated recipe and its parent evidence directory before showing this preview; it does not install a runtime, download a model, or launch a workload until `--execute` is supplied:

```bash
qualification-env/bin/python -m scripts.release.qualification_automation recipe.json --output qualification-evidence/run-001
```

After reviewing every command and ensuring required permissions are already available, opt into execution:

```bash
qualification-env/bin/python -m scripts.release.qualification_automation recipe.json --output qualification-evidence/run-001 --execute
```

Each step has a separate numbered log containing its start and finish timestamps, working directory, exact shell-escaped command, timeout, safe qualification environment overrides, subprocess output, normalized exit code, and outcome detail. A rejected workload run also prints bounded result paths, runtime/model error text, invalid-run records, and requested-versus-completed counters into that step log. `qualification-state.json` is atomically checkpointed before and after every step, and rerunning the same command resumes at the first step that has not passed. A changed recipe is rejected for an existing checkpoint; use a new evidence directory when the target, coverage, command, or timeout changes. Automation revisions use a new `-vN` evidence directory, so an older attempt remains available for audit and never needs to be deleted before retrying updated qualification code. All current targets use revision v9.

On POSIX systems the runner normalizes every evidence directory to mode `755` and every evidence file to mode `644` after each step, including failed and partial runs. If the launcher itself was invoked through `sudo`, ownership is returned to `SUDO_UID:SUDO_GID`; qualification artifacts must not remain accessible only to root.

`qualification-entry.json` is the machine-readable projection accepted by the support policy. A partial run records failed and uncompleted platform steps rather than claiming support. A completed run additionally requires `qualification-manifest.json`; the final gate refuses to emit it unless the baseline and target runtime both have complete, error-free requested evidence for every compatible workload, journals, reports, immediately verified bundles, exact runtime and model file hashes, model repositories, Python dependency inventories, host/kernel/accelerator-driver evidence, the tested Git commit and clean tracked-worktree state, cancellation/resume evidence, every executed lifecycle log, and the copied generated images for llama.cpp targets. A manifest finalization failure is reported as a failed qualification even when every lifecycle subprocess exited successfully. Qualification runs use disposable clones and never execute the legacy uninstall recipe step; deleting the clone after preserving evidence is the cleanup boundary. The manifest inventories every retained evidence file by size and SHA-256, and both result bundles contain their generated images, so moving the evidence directory does not discard the image-generation proof. Review the manifest and copy the entry into `QUALIFICATION_MATRIX` only after confirming its target identity, scope, and observed behavior.

## Human boundaries

Grant macOS privacy or security permissions and establish any required passwordless privilege before starting the unattended phase. Reboots, physical sensor plausibility, installer ownership, and the destructive scope of rollback remain human review points. The runner must not automate around an operating-system security prompt or operate on an installation whose ownership is ambiguous.
