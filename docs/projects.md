# Benchmark Projects

A local benchmark project keeps one decision workflow, one portable benchmark configuration, and optional baseline and acceptance evidence together. Projects support hardware comparison, model selection, acceptance validation, capacity planning, and regression work without introducing a server, account, or hidden project database.

## Create and open

Select **Custom** in the benchmark GUI, configure the run, then choose **New Project**. Name the project, choose its decision workflow, optionally attach an existing baseline result and an acceptance-policy JSON file, and save the `.labproject` file. **Open Project** restores its engine, tests, selected models, workload sizes, and execution settings after verifying that they are available on the current machine.

The project configuration uses the same versioned portable-preset shape as the GUI. Machine-local output and ComfyUI paths are not copied into the project; opening it retains the current machine's paths. A baseline path is intentionally local and may need to be reselected if the project file moves to another machine. An attached acceptance policy is embedded by value so later edits to a separate policy file cannot silently change the project's gate.

When a project with an acceptance policy is active, **Create Report** applies that policy automatically and lists every rule's result. A baseline is preserved as project context; local result history and baseline comparison are documented separately as they become available.

## File validation

Project schema 1 rejects unknown fields, unsupported workflows, malformed portable configurations, empty baseline paths, and invalid acceptance policies. Project files contain no credentials, Hugging Face token, output path, ComfyUI path, prompts, responses, or measurements. They remain local files unless the user explicitly transfers them.

[← Benchmark GUI](cli-reference.md#graphical-configuration) · [Back to README](../README.md) · [Acceptance Policies →](acceptance-policies.md)
