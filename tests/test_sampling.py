import pytest

from scripts.runtime.sampling import (
    BASELINE_CONTROLS, baseline_sampling_payload, baseline_sampling_profile,
    sampling_profile_payload,
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
    with pytest.raises(ValueError, match="deterministic baseline"):
        sampling_profile_payload("vllm", {"profile": "broken"})
    with pytest.raises(ValueError, match="unsupported sampling engine"):
        baseline_sampling_payload("future")
