[← Back to README](../README.md)

# Workload and engine extension contracts

These are versioned internal contracts for adding a required workload or inference engine without changing measurement semantics accidentally. They do not create dynamic plugin loading, remote code execution, or a public third-party marketplace; any new implementation remains reviewed and registered in source.

## Workload SDK v1

A workload implementation consumes an immutable `RunPlan`, resolved model identities, a capability-checked `InferenceEngine`, and a journal writer scoped to its stage. It emits only case, attempt, sample, and terminal-stage events; it does not mutate the root result document, choose output paths, start unrelated runtimes, or calculate a second authoritative aggregate. Case IDs derive from the plan's stage, model, and canonical workload-specific case key. Attempts are positive one-based ordinals, samples are positive one-based ordinals within an attempt, and every completed or invalid sample is committed before the next case starts.

The workload descriptor format is canonical finite JSON with these exact required fields:

```json
{
  "sdk_version": 1,
  "workload_id": "llm",
  "methodology_version": "4.1",
  "required_capabilities": ["generate", "model_lifecycle"],
  "case_key_fields": ["context_tokens"],
  "measurement_type": "generation-v1",
  "retry_policy": "implausible-tps-once",
  "timeout_policy": "per-sample",
  "projection_contract": "result-compatibility-v4.1#llm"
}
```

Unknown descriptor fields or versions fail validation. Identifiers use lowercase ASCII letters, digits, `_`, `.`, and `-`; lists are ordered and duplicate-free. A descriptor contains no executable, module, URL, environment, credential, or filesystem path.

A conformance vector is a directory containing `descriptor.json`, `plan.json`, `events.json`, and `expected-section.json`. The plan uses a supported run-plan schema and synthetic model identities. `events.json` is an ordered array of canonical journal event objects, including at least one valid sample, invalid sample, partial case, terminal case, and interrupted attempt when those states are supported. A conforming workload must produce the expected event identities and transitions; a conforming projection must rebuild `expected-section.json` byte-for-byte after canonical JSON encoding. Error vectors place one malformed condition in `events.json` and declare the required rejection code in `expected-error.json`. Tests use fakes only and must not start a runtime or access the network.

## Engine adapter contract v1

An adapter implements `InferenceEngine` and declares a fixed capability set before a workload starts. Capability negotiation is set inclusion: every descriptor requirement must be present or the stage is rejected as unsupported before runtime preparation. There is no fallback from one operation to a semantically different operation.

The v1 capabilities are `model_lifecycle`, `generate`, `chat`, `tools`, `embed`, `concurrency`, `context_metadata`, and `runtime_backend`. Native `llama-bench` and `llama-batched-bench` execution is not an engine capability because those workloads intentionally bind to llama.cpp tools. A capability promises the existing named request and measurement records documented in [Engines](engines.md), including timeout/partial-response behavior and explicit unavailable optional fields; it never permits returning an engine-specific response object.

Adapter conformance uses a fake runtime and the same descriptor/plan model identities. Each declared capability has a success vector, malformed-response vector, timeout vector, cancellation/cleanup vector, and—where applicable—model-load/unload and connection-recovery vector. Tests assert named measurement validity, no process ownership after cleanup, no secret/path data in events, and the absence of undeclared network or subprocess calls. An adapter cannot be registered until every capability it declares passes those vectors on each supported operating-system family; capabilities not declared must produce the common unsupported-stage result rather than an attribute error.

The adapter contract version is part of `RunPlan.execution_identity`. A behavior change to token accounting, timing boundaries, cache semantics, request shaping, recovery, or lifecycle ownership requires a new adapter-contract version or an explicit methodology boundary.

---

[← How It Works](how-it-works.md) · [Engines →](engines.md)
