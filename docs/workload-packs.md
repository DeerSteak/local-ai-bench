# Workload packs

A workload pack is a small declarative JSON document that selects an ordered subset of existing benchmark stages. It does not contain Python, shell commands, prompts, question banks, model downloads, or arbitrary configuration. This boundary makes custom selection useful without turning a result file into a plugin execution mechanism.

Built-in pack definitions live in `scripts/workloads/workload_packs.py`. `core-v1` selects the commercially credible cross-platform workloads, while `native-crosscheck-v1` selects llama.cpp's native cross-checks. A pack identity consists of its ID, positive integer version, complete content, and SHA-256 digest. Published content is immutable: any stage, compatibility, or metadata change requires a new version and therefore a new digest.

Compatibility is an explicit allowlist of Local AI Bench application versions. Version 1 packs support application version 4.1 only. Loading rejects missing or unknown fields, duplicate or unknown stages, unsupported schemas, invalid origins, and incompatible application versions rather than guessing.

To author a custom pack, copy this example to a local `*.labpack.json` file, choose only stage keys documented in the CLI reference, set `origin` to `custom`, and validate it with `load_custom_pack()` before associating it with a project. Custom packs remain local in this first format and cannot define new executable workloads.

```json
{
  "schema_version": 1,
  "id": "focused-llm",
  "version": 1,
  "stages": ["llm", "conv"],
  "application_versions": ["4.1"],
  "origin": "custom"
}
```

Adding genuinely new workload logic remains a reviewed code change with tests, methodology definitions, licensing provenance, and a new application release. A future third-party catalog must define signing, review, revocation, and compatibility policy before it accepts executable extensions.
