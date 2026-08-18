[← Back to README](../README.md)

# Platform qualification automation

Platform qualification uses a reviewed JSON recipe and a resumable runner. The runner automates lifecycle evidence; it does not turn a smoke workload into a full-catalog performance claim.

## Scope

A lifecycle recipe runs one deliberately small representative workload, normally one xsmall LLM model for the selected engine. It verifies install, discovery, a valid run, cancellation, resume, report generation, bundle export, upgrade, rollback, and uninstall. The recipe records its workload and model coverage so the evidence cannot imply that every model was exercised.

Use a separate performance qualification when a claim depends on catalog-wide compatibility, throughput, accuracy, image generation, embeddings, or model-specific behavior.

## Recipe

Copy [`samples/qualification_recipe_example.json`](../samples/qualification_recipe_example.json) outside the repository or into a private evidence workspace and replace every placeholder with an argv command. Commands are JSON arrays and are executed directly without a shell. Give install, upgrade, rollback, and uninstall commands an isolated qualification prefix; never point them at the normal installation or model store. The recipe may record only the allowlisted non-secret runtime environment fields; credentials such as a Hugging Face token must be inherited at execution time and never written into the recipe or evidence.

The bundled install step uses `scripts.release.qualification_install` to install the selected llama.cpp or vLLM runtime and download only the recorded smoke model beneath a disposable repository clone. It previews by default and requires both `--execute` and `--confirm-isolated-root` before changing that clone. Runtime installation still obeys the platform support checks; an unsupported vLLM combination fails instead of falling back to another backend. vLLM installation additionally requires `--runtime-version` with the exact wheel identity, including a ROCm local-version suffix such as `0.27.1+rocm723`; qualification never installs a floating latest build. A vLLM recipe sets `HF_HOME` to the disposable clone's `qualification-vllm-cache` directory so installation and the later smoke run resolve the same weights.

On a fresh machine, Python 3.11 or newer and Git are bootstrap prerequisites because the qualification code cannot run before the repository and interpreter exist. Create `bench-env`, install `requirements.txt`, and then let the recipe install the engine and smoke model. Host package-manager changes require an explicit platform-administrator action; credentials and interactive operating-system permissions are never bypassed by the qualification runner.

The cancellation command is the only command that may define `interrupt_after_seconds`. The runner launches it in its own process group, sends the platform interrupt signal after the delay, and accepts only the declared exit codes. Configure the delay so the smoke workload has entered a recoverable stage rather than merely started its launcher.

## Preview and execution

Preview is the default and creates no directories or files:

```bash
bench-env/bin/python -m scripts.release.qualification_automation recipe.json --output qualification-evidence/run-001
```

After reviewing every command and ensuring required permissions are already available, opt into execution:

```bash
bench-env/bin/python -m scripts.release.qualification_automation recipe.json --output qualification-evidence/run-001 --execute
```

Each step has a separate numbered log. `qualification-state.json` is atomically checkpointed before and after every step, and rerunning the same command resumes at the first step that has not passed. A changed recipe is rejected for an existing checkpoint; use a new evidence directory when the target, coverage, command, or timeout changes.

`qualification-entry.json` is the machine-readable projection accepted by the support policy. A partial run records failed and uncompleted lifecycle steps rather than claiming support. Review the logs and copy the entry into `QUALIFICATION_MATRIX` only after confirming its target identity, evidence paths, scope, and observed behavior.

## Human boundaries

Grant macOS privacy or security permissions and establish any required passwordless privilege before starting the unattended phase. Reboots, physical sensor plausibility, installer ownership, and the destructive scope of rollback or uninstall remain human review points. The runner must not automate around an operating-system security prompt or operate on an installation whose ownership is ambiguous.
