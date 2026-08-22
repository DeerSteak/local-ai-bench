import pytest

from scripts.runtime.mtp import (
    mtp_mode_states, mtp_pass_label, mtp_tests, native_mtp_config, native_mtp_models,
)
from scripts.workloads.models import LLM_MODELS


def test_catalog_marks_every_confirmed_native_vllm_mtp_artifact():
    supported = {
        model["tag"] for model in native_mtp_models(LLM_MODELS, "vllm")
    }
    assert supported == {
        "qwen3.5:4b-q4_K_M",
        "qwen3.5:9b-q4_K_M",
        "qwen3.8:27b-ud-q4_K_M",
        "nemotron3.5-lightning:30b-a3b-ud-q4_K_M",
        "nemotron-3-super:120b",
    }
    assert not native_mtp_models(LLM_MODELS, "llamacpp")


@pytest.mark.parametrize("value", [None, {}, {"vllm": {}}, {"vllm": {"num_speculative_tokens": 0}},
                                   {"vllm": {"num_speculative_tokens": True}}])
def test_native_mtp_config_rejects_incomplete_or_invalid_metadata(value):
    assert native_mtp_config({"native_mtp": value}, "vllm") is None


def test_native_mtp_config_returns_a_defensive_engine_specific_payload():
    source = {"vllm": {"num_speculative_tokens": 2, "unrelated": "ignored"}}
    assert native_mtp_config({"native_mtp": source}, "vllm") == {
        "num_speculative_tokens": 2,
    }
    assert native_mtp_config({"native_mtp": source}, "llamacpp") is None


def test_mtp_mode_states_expand_comparison_mode():
    assert mtp_mode_states("off") == (False,)
    assert mtp_mode_states("on") == (True,)
    assert mtp_mode_states("both") == (False, True)
    with pytest.raises(ValueError, match="unknown MTP mode"):
        mtp_mode_states("sometimes")


def test_mtp_on_keeps_only_server_backed_generation_workloads():
    tests = ["llm", "conv", "llamabench", "vllmbench", "emb", "mcq", "conc_chat", "img"]
    assert mtp_tests(tests, False) == tests
    assert mtp_tests(tests, True) == ["llm", "conv", "mcq", "conc_chat"]


def test_mtp_pass_label_distinguishes_progress_identity():
    assert mtp_pass_label("vllm", False) == "vllm · MTP off"
    assert mtp_pass_label("vllm", True) == "vllm · MTP on"
