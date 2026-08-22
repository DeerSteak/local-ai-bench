from pathlib import Path

import pytest

from scripts.release.model_catalog_screen import (
    ScreenSpec, build_screen_spec, candidate_import_matches, candidate_record,
    compatibility_screen_errors, interrupt_ready, select_exact_variant,
    pipeline_asset_target, screen_image_artifacts,
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
                "configuration": {
                    "context_tokens": 131072,
                    "publisher_sampling": {
                        "do_sample": True, "temperature": 1.0,
                        "top_k": 20, "top_p": 0.95,
                    },
                },
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
        baseline_sampling_profile("llamacpp"),
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

    publisher = build_screen_spec(
        record, "llamacpp", tmp_path, python_executable="python",
        publisher_sampling=True,
    )
    assert publisher.sampling_profile["profile"] == (
        "publisher-recommended-v1:candidate"
    )
    assert publisher.output_path.parent.name == "publisher"
    assert publisher.command[-2:] == (
        "--publisher-sampling-profile",
        str(publisher.output_path.with_name("publisher-sampling.json")),
    )


def test_screen_plan_refuses_blocked_and_unimplemented_candidates(tmp_path):
    with pytest.raises(ValueError, match="blocked reason"):
        build_screen_spec(candidate(status="blocked"), "llamacpp", tmp_path)
    with pytest.raises(ValueError, match="supported fixed ComfyUI workflow"):
        build_screen_spec(candidate(family="image"), "llamacpp", tmp_path)


def test_z_image_screen_uses_fixed_normal_workload_spec(tmp_path):
    record = candidate(family="image")
    record["id"] = "z-image-turbo"
    record["name"] = "Z-Image Turbo"
    record["sources"]["pipeline"] = [{
        "repo": "Comfy-Org/z_image_turbo", "revision": "c" * 40,
        "files": [
            {"name": "split_files/text_encoders/qwen_3_4b.safetensors", "sha256": "1" * 64},
            {"name": "split_files/diffusion_models/z_image_turbo_bf16.safetensors", "sha256": "2" * 64},
            {"name": "split_files/vae/ae.safetensors", "sha256": "3" * 64},
        ],
    }]
    screen = build_screen_spec(record, "llamacpp", tmp_path)
    assert screen.tag == "z-image-turbo"
    assert screen.image_model is not None
    assert screen.image_model == {
        "audit_candidate": True,
        "artifact_digest": screen.image_model["artifact_digest"],
        "label": "Z-Image Turbo", "short": "z-image-turbo", "tier": "medium",
        "checkpoint": "z_image_turbo_bf16.safetensors",
        "checkpoint_folder": "diffusion_models", "workflow": "z_image",
        "steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple",
    }
    assert "--audit-image-model" in screen.command
    assert screen.files == (
        "split_files/text_encoders/qwen_3_4b.safetensors",
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "split_files/vae/ae.safetensors",
    )
    assert pipeline_asset_target(screen.files[0], tmp_path) == (
        tmp_path / "text_encoders" / "qwen_3_4b.safetensors"
    )
    with pytest.raises(ValueError, match="invalid path"):
        pipeline_asset_target("checkpoints/escape.safetensors", tmp_path)


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


def test_complete_publisher_screen_requires_its_distinct_methodology_and_sampler(tmp_path):
    screen_spec = build_screen_spec(
        candidate(), "llamacpp", tmp_path, publisher_sampling=True,
    )
    result = complete_result(screen_spec)
    result["run"]["plan"]["effective_config"].update({
        "methodology_profile": "publisher-v1",
        "sampling_profile": screen_spec.sampling_profile,
    })
    assert compatibility_screen_errors(result, screen_spec) == []


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


def test_image_screen_requires_recovered_stage_and_measured_resolution():
    screen_spec = spec("image")
    result = {
        "run": {
            "status": "complete", "recovery_history": [{"status": "interrupted"}],
            "stages": {"img": {"status": "complete"}},
            "plan": {"effective_config": {"methodology_profile": "neutral-v2"}},
        },
        "preflight": {"models": {}},
        "images": {screen_spec.tag: {
            "resolutions": {"1024x1024": {"n_runs": 1}},
        }},
    }
    assert interrupt_ready(result, screen_spec)
    assert compatibility_screen_errors(result, screen_spec) == []
    result["images"][screen_spec.tag]["resolutions"] = {}
    assert compatibility_screen_errors(result, screen_spec) == [
        "image measurement evidence is missing",
    ]


def test_image_screen_report_hashes_every_generated_resolution(tmp_path):
    screen_spec = ScreenSpec(
        "z-image-turbo", "Z-Image Turbo", "llamacpp", "z-image-turbo", "image",
        "owner/model", "a" * 40, (), None, tmp_path / "result.json", (),
        baseline_sampling_profile("llamacpp"),
    )
    image_dir = tmp_path / "images_result"
    image_dir.mkdir()
    (image_dir / "z-image-turbo_1024x1024.png").write_bytes(b"png")
    result = {"images": {screen_spec.tag: {
        "resolutions": {"1024x1024": {"n_runs": 1}, "1536x1536": {"n_runs": 1}},
    }}}
    artifacts, errors = screen_image_artifacts(result, screen_spec)
    assert artifacts[0]["resolution"] == "1024x1024"
    assert artifacts[0]["size"] == 3
    assert errors == ["generated image is missing: 1536x1536"]
