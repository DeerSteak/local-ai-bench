# Installation maintenance

Maintenance is deliberately split between project-owned files and external/system software. Local AI Bench may inspect or remove only its own repository-managed components. It never removes a system llama.cpp installation, a user-managed ComfyUI installation, Homebrew, Python, GPU drivers, or operating-system packages.

## Repair

`installation_health()` performs a read-only check for the project environment, requirements manifest, and setup/benchmark launchers. Repair re-runs the ordinary setup flow after showing the same prerequisite, runtime, model, disk, credential, and ComfyUI choices; setup reuses valid existing components and downloads only missing managed artifacts. It must not delete results, models, credentials, presets, projects, or configuration as a repair shortcut.

## Upgrade and rollback

Version 6.0-pre7 has no automatic updater. A preview upgrade is a separately downloaded source release applied only after the user backs up projects/results and reviews its release notes and compatibility axes. The old directory remains the rollback copy until the new build passes setup health and a verification run. Models and imported result bundles can be reused through explicit paths, but executable environments are rebuilt from the target release rather than shared across versions.

Signed in-place upgrade and rollback remain stable-release blockers. They require signed packages and manifests, downgrade protection, staged channels, migration dry runs, power/network interruption tests, and a tested last-known-good restoration path. The application must not present manual source replacement as a commercially supported automatic update.

## Uninstall

`build_uninstall_plan()` previews exact repository-owned paths. The default removes only `bench-env/` and managed runtime source/install directories (`ComfyUI/` and `llama.cpp/` when present). It preserves downloaded models, results, `local_ai_bench_config.json`, `hf.txt`, projects, presets, and the source tree. Models, results, and credentials require separate explicit choices.

`execute_uninstall_plan()` accepts only immediate children from a fixed managed-name allowlist and requires the exact typed confirmation `REMOVE LOCAL AI BENCH COMPONENTS`. It rejects unrelated directories and never follows a plan target outside the validated repository root. The GUI/packaged maintenance surface should present this plan and preservation list before enabling removal.

Removing the marked Local AI Bench block from a user-managed ComfyUI configuration is a separate reversible cleanup action; it must preserve every other line and should occur only after previewing that exact file. Until that action has a packaged UI and tests, uninstall leaves the harmless model-path entry in place and tells the user where it is.
