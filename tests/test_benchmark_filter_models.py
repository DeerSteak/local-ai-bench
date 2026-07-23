from benchmark import filter_models_by_pattern
from models import LLM_MODELS


def test_no_patterns_returns_models_unchanged():
    assert filter_models_by_pattern(LLM_MODELS, None) == LLM_MODELS
    assert filter_models_by_pattern(LLM_MODELS, []) == LLM_MODELS


def test_exact_tag_matches_only_that_model():
    result = filter_models_by_pattern(LLM_MODELS, ["qwen3.5:4b-q4_K_M"])
    assert [m["tag"] for m in result] == ["qwen3.5:4b-q4_K_M"]


def test_wildcard_matches_every_tag_with_that_prefix():
    result = filter_models_by_pattern(LLM_MODELS, ["llama*"])
    tags = {m["tag"] for m in result}
    assert tags == {
        "llama3.3:70b-instruct-q4_K_M",
    }


def test_wildcard_is_case_sensitive():
    # Tags are always lowercase — an uppercase pattern shouldn't match.
    assert filter_models_by_pattern(LLM_MODELS, ["LLAMA*"]) == []


def test_multiple_patterns_are_unioned_without_duplicates():
    result = filter_models_by_pattern(LLM_MODELS, ["qwen3.5:4b*", "qwen*"])
    tags = [m["tag"] for m in result]
    assert tags.count("qwen3.5:4b-q4_K_M") == 1
    assert set(tags) == {model["tag"] for model in LLM_MODELS
                         if model["tag"].startswith("qwen")}


def test_no_match_returns_empty_list():
    assert filter_models_by_pattern(LLM_MODELS, ["nonexistent-model*"]) == []


def test_preserves_original_model_order():
    result = filter_models_by_pattern(LLM_MODELS, ["*"])
    assert result == LLM_MODELS
