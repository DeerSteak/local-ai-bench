from pathlib import Path

import pytest

from scripts.release.model_catalog_screen import (
    ScreenSpec, build_screen_spec, candidate_import_matches, candidate_record,
    compatibility_screen_errors, interrupt_ready, select_exact_variant,
)
from scripts.runtime.sampling import baseline_sampling_profile
from scripts.setup.model_import import ImportVariant, RepositoryInspection


def candidate(*, status="source_ready", family="llm"):
    artifact = {"files": ["model.gguf"], "kind": "gguf"}
    return {
        "id": "candidate", "name": "Candidate", "family": family,
        "status": status, "reasons": ["blocked reason"] if status != "source_ready" else [],
        "sources": {
            "upstream": {
                "repo": "owner/model", "revision": "a" * 40,
                "artifact": {"files": ["model.safetensors"], "kind": "safetensors"},
                "configuration": {"context_tokens": 131072},
            },
            "gguf": {
                "repo": "owner/model-gguf", "revision": "b" * 40, "artifact": artifact,
            },
        },
    }


def spec(family="llm"):
    return ScreenSpec(
        "candidate", "Candidate", "llamacpp", "audit-candidate", family,
        "owner/model-gguf", "b" * 40, ("model.gguf",), 131072,
        Path("/tmp/result.json"), ("python", "-m", "scripts.app.benchmark"),
    )


def complete_result(screen_spec):
    tag = screen_spec.tag
    sample = {"valid_runs": 1}
    return {
        "run": {
            "status": "complete",
            "recovery_history": [{"status": "interrupted"}],
            "stages": {"llm": {"status": "complete"}, "conv": {"status": "complete"}},
            "plan": {"effective_config": {
                "methodology_profile": "neutral-v2",
                "sampling_profile": baseline_sampling_profile(screen_spec.engine),
            }},
        },
        "preflight": {"models": {tag: {
            "status": "passed",
            "checks": [{"name": "formatting_probe", "status": "passed"}],
        }}},
        "llm": {tag: {"2K": sample, "32K": sample}},
        "llm_conversation": {tag: {"2K": sample, "32K": sample}},
    }


def test_candidate_lookup_and_screen_plan_are_exact_and_side_effect_free(tmp_path):
    record = candidate()
    audit = {"candidates": [record]}
    assert candidate_record(audit, "candidate") is record
    with pytest.raises(ValueError, match="unknown"):
        candidate_record(audit, "missing")

    llama = build_screen_spec(record, "llamacpp", tmp_path, python_executable="python")
    assert llama.repo == "owner/model-gguf"
    assert llama.files == ("model.gguf",)
    assert llama.output_path == tmp_path / "candidate" / "llamacpp" / ("b" * 12) / "result.json"
    assert llama.command[:3] == ("python", "-m", "scripts.app.benchmark")
    assert ("--tests", "llm", "conv") == llama.command[5:8]
    assert llama.command[llama.command.index("--max-prompt-tokens") + 1] == "32768"
    assert "--force-all" in llama.command

    vllm = build_screen_spec(record, "vllm", tmp_path, python_executable="python")
    assert vllm.repo == "owner/model"
    assert vllm.files == ("model.safetensors",)
    assert vllm.command[-1] == "--ack-experimental-engine"


def test_screen_plan_refuses_blocked_and_unimplemented_candidates(tmp_path):
    with pytest.raises(ValueError, match="blocked reason"):
        build_screen_spec(candidate(status="blocked"), "llamacpp", tmp_path)
    with pytest.raises(ValueError, match="ComfyUI workflow"):
        build_screen_spec(candidate(family="image"), "llamacpp", tmp_path)


def test_exact_variant_selection_never_falls_back_to_another_quantization():
    wanted = ImportVariant("wanted", "wanted", ("wanted.gguf",), 10)
    other = ImportVariant("other", "other", ("other.gguf",), 5)
    inspection = RepositoryInspection(
        "owner/repo", "revision", (other, wanted), None, False,
    )
    assert select_exact_variant(inspection, "llamacpp", ("wanted.gguf",)) is wanted
    with pytest.raises(ValueError, match="no longer matches"):
        select_exact_variant(inspection, "llamacpp", ("missing.gguf",))


def test_existing_import_identity_handles_llamacpp_files_and_vllm_cache_registry_shape(tmp_path):
    llama = spec()
    assert candidate_import_matches({
        "repo": llama.repo, "revision": llama.revision,
        "format": "gguf", "files": ["model.gguf"],
    }, llama)
    assert not candidate_import_matches({
        "repo": llama.repo, "revision": "changed",
        "format": "gguf", "files": ["model.gguf"],
    }, llama)

    vllm = build_screen_spec(candidate(), "vllm", tmp_path)
    assert candidate_import_matches({
        "repo": vllm.repo, "revision": vllm.revision,
        "format": "safetensors", "files": [],
    }, vllm)


def test_interrupt_waits_for_durable_llm_evidence_and_embedding_stage_start():
    llm = spec()
    assert not interrupt_ready({"llm": {llm.tag: {"2K": {"valid_runs": 0}}}}, llm)
    assert interrupt_ready({"llm": {llm.tag: {"2K": {"valid_runs": 1}}}}, llm)
    embedding = spec("embedding")
    assert not interrupt_ready({"run": {"stages": {}}}, embedding)
    assert interrupt_ready(
        {"run": {"stages": {"emb": {"status": "running"}}}}, embedding,
    )


def test_complete_screen_requires_recovery_preflight_sampling_and_both_context_paths():
    screen_spec = spec()
    result = complete_result(screen_spec)
    assert compatibility_screen_errors(result, screen_spec) == []

    result["run"]["recovery_history"] = []
    result["run"]["plan"]["effective_config"]["sampling_profile"]["profile"] = "changed"
    result["preflight"]["models"][screen_spec.tag]["checks"][0]["status"] = "empty"
    del result["llm_conversation"][screen_spec.tag]["32K"]
    assert compatibility_screen_errors(result, screen_spec) == [
        "interrupt/resume evidence is missing",
        "resolved sampler identity is missing or incorrect",
        "chat formatting probe did not pass",
        "conversation 32K evidence is missing",
    ]


def test_embedding_screen_requires_one_valid_measurement_without_sampler_claim():
    screen_spec = spec("embedding")
    result = {
        "run": {
            "status": "complete", "recovery_history": [{"status": "interrupted"}],
            "stages": {"emb": {"status": "complete"}},
            "plan": {"effective_config": {"methodology_profile": "neutral-v2"}},
        },
        "preflight": {"models": {screen_spec.tag: {"status": "passed", "checks": []}}},
        "embeddings": {screen_spec.tag: {"valid_runs": 1}},
    }
    assert compatibility_screen_errors(result, screen_spec) == []
