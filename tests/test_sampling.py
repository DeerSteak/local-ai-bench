import json

import pytest

from scripts.runtime.sampling import (
    BASELINE_CONTROLS, baseline_sampling_payload, baseline_sampling_profile,
    load_publisher_sampling_profile, publisher_sampling_profile, sampling_profile_payload,
)


def test_baseline_semantics_pin_every_shared_logit_control():
    assert BASELINE_CONTROLS == {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 0,
        "logit_bias": {},
    }


def test_engine_payloads_resolve_equivalent_shared_controls():
    vllm = baseline_sampling_payload("vllm")
    llamacpp = baseline_sampling_payload("llamacpp")
    for key in (
        "temperature", "top_k", "top_p", "min_p", "presence_penalty",
        "frequency_penalty", "seed",
    ):
        assert llamacpp[key] == vllm[key]
    assert llamacpp["repeat_penalty"] == vllm["repetition_penalty"] == 1.0
    assert llamacpp["logit_bias"] == [] and vllm["logit_bias"] == {}


def test_llamacpp_payload_disables_engine_specific_samplers():
    payload = baseline_sampling_payload("llamacpp")
    assert payload | {
        "typical_p": 1.0,
        "dry_multiplier": 0.0,
        "xtc_probability": 0.0,
        "top_n_sigma": -1.0,
        "dynatemp_range": 0.0,
        "mirostat": 0,
    } == payload


def test_sampling_profiles_are_fresh_and_reject_unknown_engines():
    first = baseline_sampling_profile("vllm")
    first["semantic_controls"]["temperature"] = 9
    first["semantic_controls"]["logit_bias"]["4"] = 1
    assert baseline_sampling_profile("vllm")["semantic_controls"]["temperature"] == 0.0
    assert baseline_sampling_profile("vllm")["semantic_controls"]["logit_bias"] == {}


def test_publisher_profile_is_source_pinned_fully_resolved_and_engine_specific():
    controls = {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    llama = publisher_sampling_profile(
        "llamacpp", name="qwen", repo="owner/model", revision="a" * 40,
        controls=controls,
    )
    vllm = publisher_sampling_profile(
        "vllm", name="qwen", repo="owner/model", revision="a" * 40,
        controls=controls,
    )
    assert llama["profile"] == "publisher-recommended-v1:qwen"
    assert llama["source"] == {
        "repo": "owner/model", "revision": "a" * 40,
        "path": "generation_config.json",
        "controls_sha256": "f94223a7a0e4bd507fecd4e0770189b22f88067d9c7882288a426292fabb9bd6",
    }
    assert llama["publisher_controls"] == controls
    assert llama["semantic_controls"]["min_p"] == 0.0
    assert llama["engine_controls"]["repeat_penalty"] == 1.0
    assert "repetition_penalty" not in llama["engine_controls"]
    assert vllm["engine_controls"]["repetition_penalty"] == 1.0
    assert "do_sample" not in vllm["engine_controls"]
    assert sampling_profile_payload("vllm", vllm) == vllm["engine_controls"]
    vllm["source"]["controls_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        sampling_profile_payload("vllm", vllm)


def test_publisher_profile_rejects_missing_or_unsupported_source_controls():
    kwargs = {"name": "model", "repo": "owner/model", "revision": "a" * 40}
    with pytest.raises(ValueError, match="empty"):
        publisher_sampling_profile("llamacpp", controls={}, **kwargs)
    with pytest.raises(ValueError, match="unsupported"):
        publisher_sampling_profile("llamacpp", controls={"epsilon_cutoff": 1}, **kwargs)
    with pytest.raises(ValueError, match="disables sampling"):
        publisher_sampling_profile("llamacpp", controls={"do_sample": False}, **kwargs)
    with pytest.raises(ValueError, match="resolved engine controls"):
        sampling_profile_payload("llamacpp", {"profile": "broken"})


def test_publisher_profile_file_requires_versioned_exact_source(tmp_path):
    path = tmp_path / "publisher.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "name": "qwen",
        "source": {"repo": "owner/model", "revision": "a" * 40},
        "controls": {"do_sample": True, "temperature": 1.0},
    }), encoding="utf-8")
    assert load_publisher_sampling_profile(path, "vllm")["source"]["revision"] == "a" * 40

    path.write_text(json.dumps({
        "schema_version": 1, "name": "qwen", "source": {"repo": "owner/model"},
        "controls": {"temperature": 1.0},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="exact source"):
        load_publisher_sampling_profile(path, "vllm")
    with pytest.raises(ValueError, match="unsupported sampling engine"):
        baseline_sampling_payload("future")
