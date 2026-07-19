from benchmark import filter_models_by_pattern
from models import LLM_MODELS


def test_no_patterns_returns_models_unchanged():
    assert filter_models_by_pattern(LLM_MODELS, None) == LLM_MODELS
    assert filter_models_by_pattern(LLM_MODELS, []) == LLM_MODELS


def test_exact_tag_matches_only_that_model():
    result = filter_models_by_pattern(LLM_MODELS, ["phi4-mini"])
    assert [m["tag"] for m in result] == ["phi4-mini"]


def test_wildcard_matches_every_tag_with_that_prefix():
    result = filter_models_by_pattern(LLM_MODELS, ["llama*"])
    tags = {m["tag"] for m in result}
    assert tags == {
        "llama3.2:3b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.3:70b-instruct-q4_K_M",
        "llama4:16x17b",
    }


def test_wildcard_is_case_sensitive():
    # Tags are always lowercase — an uppercase pattern shouldn't match.
    assert filter_models_by_pattern(LLM_MODELS, ["LLAMA*"]) == []


def test_multiple_patterns_are_unioned_without_duplicates():
    result = filter_models_by_pattern(LLM_MODELS, ["llama3.2*", "llama*"])
    tags = [m["tag"] for m in result]
    assert tags.count("llama3.2:3b-instruct-q4_K_M") == 1
    assert set(tags) == {
        "llama3.2:3b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.3:70b-instruct-q4_K_M",
        "llama4:16x17b",
    }


def test_no_match_returns_empty_list():
    assert filter_models_by_pattern(LLM_MODELS, ["nonexistent-model*"]) == []


def test_preserves_original_model_order():
    result = filter_models_by_pattern(LLM_MODELS, ["*"])
    assert result == LLM_MODELS
