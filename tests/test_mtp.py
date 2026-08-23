import pytest

from scripts.runtime.mtp import (
    active_mtp_configurations, expand_mtp_passes, mtp_mode_states, mtp_pass_label,
    mtp_progress_names, mtp_selection_error, mtp_tests, native_mtp_config,
    native_mtp_models,
)
from scripts.workloads.models import LLM_MODELS


def test_catalog_marks_every_confirmed_native_mtp_artifact():
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
    assert {model["tag"] for model in native_mtp_models(LLM_MODELS, "llamacpp")} == {
        "qwen3.5:4b-q4_K_M",
        "qwen3.5:9b-q4_K_M",
        "qwen3.8:27b-ud-q4_K_M",
        "nemotron3.5-lightning:30b-a3b-ud-q4_K_M",
    }
    catalog = {model["tag"]: model for model in LLM_MODELS}
    tags = (
        "qwen3.5:4b-q4_K_M", "qwen3.5:9b-q4_K_M",
        "qwen3.8:27b-ud-q4_K_M",
        "nemotron3.5-lightning:30b-a3b-ud-q4_K_M",
    )
    configs = {tag: native_mtp_config(catalog[tag], "vllm") for tag in tags}
    assert all(config is not None for config in configs.values())
    assert {tag: config["num_speculative_tokens"] for tag, config in configs.items()
            if config is not None} == {
        "qwen3.5:4b-q4_K_M": 3,
        "qwen3.5:9b-q4_K_M": 3,
        "qwen3.8:27b-ud-q4_K_M": 3,
        "nemotron3.5-lightning:30b-a3b-ud-q4_K_M": 1,
    }


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


def test_native_mtp_config_returns_validated_separate_draft_metadata():
    source = {"llamacpp": {
        "num_speculative_tokens": 3,
        "draft_repo": "owner/model",
        "draft_file": "MTP/draft.gguf",
        "draft_download_size": "~1 GB",
    }}
    assert native_mtp_config({"native_mtp": source}, "llamacpp") == {
        "num_speculative_tokens": 3,
        "draft_repo": "owner/model",
        "draft_file": "MTP/draft.gguf",
    }


def test_native_mtp_config_does_not_forward_llamacpp_draft_fields_to_vllm():
    source = {"vllm": {
        "num_speculative_tokens": 1,
        "draft_repo": "ignored", "draft_file": "ignored.gguf",
    }}
    assert native_mtp_config({"native_mtp": source}, "vllm") == {
        "num_speculative_tokens": 1,
    }


@pytest.mark.parametrize("config", [
    {"num_speculative_tokens": 3, "draft_repo": "owner/model"},
    {"num_speculative_tokens": 3, "draft_file": "draft.gguf"},
    {"num_speculative_tokens": 3, "draft_repo": "", "draft_file": "draft.gguf"},
])
def test_native_mtp_config_rejects_incomplete_separate_draft_metadata(config):
    assert native_mtp_config({"native_mtp": {"llamacpp": config}}, "llamacpp") is None


def test_active_mtp_configurations_record_engine_specific_tokens_and_predictor_mode():
    models = [
        {"tag": "embedded", "native_mtp": {
            "llamacpp": {"num_speculative_tokens": 3},
            "vllm": {"num_speculative_tokens": 1},
        }},
        {"tag": "separate", "native_mtp": {"llamacpp": {
            "num_speculative_tokens": 2,
            "draft_repo": "owner/model", "draft_file": "draft.gguf",
        }}},
        {"tag": "unsupported"},
    ]
    assert active_mtp_configurations(models, "llamacpp", True) == {
        "embedded": {"num_speculative_tokens": 3, "predictor": "embedded"},
        "separate": {"num_speculative_tokens": 2, "predictor": "separate"},
    }
    assert active_mtp_configurations(models, "vllm", True) == {
        "embedded": {"num_speculative_tokens": 1, "predictor": "embedded"},
    }
    assert active_mtp_configurations(models, "llamacpp", False) == {}


def test_active_mtp_configurations_deduplicate_identical_models_and_reject_bad_identity():
    capable = {"tag": "same", "native_mtp": {
        "llamacpp": {"num_speculative_tokens": 3},
    }}
    assert active_mtp_configurations([capable, capable], "llamacpp", True) == {
        "same": {"num_speculative_tokens": 3, "predictor": "embedded"},
    }
    with pytest.raises(ValueError, match="requires a tag"):
        active_mtp_configurations([
            {"native_mtp": {"llamacpp": {"num_speculative_tokens": 3}}},
        ], "llamacpp", True)


def test_active_mtp_configurations_reject_conflicting_duplicate_tags():
    models = [
        {"tag": "same", "native_mtp": {"llamacpp": {"num_speculative_tokens": 1}}},
        {"tag": "same", "native_mtp": {"llamacpp": {"num_speculative_tokens": 3}}},
    ]
    with pytest.raises(ValueError, match="conflicting MTP configuration"):
        active_mtp_configurations(models, "llamacpp", True)


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


def test_mtp_on_rejects_non_mtp_workloads_instead_of_dropping_them():
    capable = {"tag": "capable", "native_mtp": {"llamacpp": {
        "num_speculative_tokens": 1,
    }}}
    assert mtp_selection_error(
        ["llamacpp"], "on", [capable], ["llm", "img", "emb", "llamabench"],
    ) == (
        "--mtp on cannot run non-MTP workloads: img, emb, llamabench; "
        "use --mtp both to run them once in the baseline pass"
    )
    assert mtp_selection_error(
        ["llamacpp"], "both", [capable], ["llm", "img"],
    ) is None


def test_mtp_on_requires_support_in_the_selected_models():
    assert mtp_selection_error(
        ["llamacpp", "vllm"], "on", [{"tag": "plain"}], ["llm"],
    ) == (
        "--mtp on requires a selected model with cataloged native MTP support for every "
        "selected engine; missing: llamacpp, vllm"
    )


def test_mtp_on_rejects_an_engine_that_would_be_silently_dropped():
    vllm_only = {"tag": "capable", "native_mtp": {
        "vllm": {"num_speculative_tokens": 1},
    }}
    assert mtp_selection_error(
        ["llamacpp", "vllm"], "on", [vllm_only], ["llm"],
    ) == (
        "--mtp on requires a selected model with cataloged native MTP support for every "
        "selected engine; missing: llamacpp"
    )


def test_mtp_both_rejects_a_baseline_only_selection():
    assert mtp_selection_error(
        ["llamacpp"], "both", [{"tag": "plain"}], ["llm"],
    ) == (
        "--mtp both requires a selected model with cataloged native MTP support; "
        "use --mtp off for a baseline-only run"
    )


def test_mtp_both_requires_a_workload_that_can_run_the_on_pass():
    capable = {"tag": "capable", "native_mtp": {
        "llamacpp": {"num_speculative_tokens": 1},
    }}
    assert mtp_selection_error(
        ["llamacpp"], "both", [capable], ["img", "emb"],
    ) == (
        "--mtp both requires at least one server-backed text workload; "
        "use --mtp off for a baseline-only run"
    )


def test_mtp_pass_label_distinguishes_progress_identity():
    assert mtp_pass_label("vllm", False) == "vllm · MTP off"
    assert mtp_pass_label("vllm", True) == "vllm · MTP on"


def test_both_mode_keeps_full_baseline_and_filters_native_mtp_pass():
    capable = {"tag": "capable", "native_mtp": {"vllm": {"num_speculative_tokens": 1}}}
    unsupported = {"tag": "unsupported"}
    scope = {
        "name": "vllm", "tests": ["llm", "emb", "conc_chat", "img"],
        "llm_models": [capable, unsupported],
        "concurrency_models": [capable, unsupported],
        "embedding_models": [{"tag": "embed"}],
    }
    passes = expand_mtp_passes([scope], "both")
    assert [(item["mtp_enabled"], item["tests"]) for item in passes] == [
        (False, ["llm", "emb", "conc_chat", "img"]),
        (True, ["llm", "conc_chat"]),
    ]
    assert [model["tag"] for model in passes[1]["llm_models"]] == ["capable"]
    assert [model["tag"] for model in passes[1]["concurrency_models"]] == ["capable"]
    assert passes[1]["progress_name"] == "vllm · MTP on"


def test_on_mode_drops_engines_and_workloads_without_native_mtp_support():
    scope = {
        "name": "llamacpp", "tests": ["llm", "llamabench", "img"],
        "llm_models": [{"tag": "plain"}], "concurrency_models": [],
    }
    assert expand_mtp_passes([scope], "on") == []


def test_on_mode_can_run_only_a_capable_concurrency_scope():
    capable = {"tag": "capable", "native_mtp": {"vllm": {"num_speculative_tokens": 1}}}
    scope = {
        "name": "vllm", "tests": ["llm", "conc_tool"],
        "llm_models": [], "concurrency_models": [capable],
    }
    assert expand_mtp_passes([scope], "on")[0]["tests"] == ["conc_tool"]


def test_progress_names_include_only_engines_with_an_mtp_pass():
    models = [{
        "tag": "capable", "native_mtp": {"vllm": {"num_speculative_tokens": 1}},
    }]
    assert mtp_progress_names(["llamacpp", "vllm"], "both", models) == [
        "llamacpp · MTP off", "vllm · MTP off", "vllm · MTP on",
    ]
