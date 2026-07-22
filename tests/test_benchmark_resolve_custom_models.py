from benchmark import resolve_custom_models, sanitize_tag_to_short
from models import LLM_MODELS


def test_catalog_pattern_behaves_like_filter_models_by_pattern():
    result = resolve_custom_models(["llama*"], LLM_MODELS, installed_tags=[])
    tags = {m["tag"] for m in result}
    assert tags == {
        "llama3.2:3b-instruct-q4_K_M",
        "llama3.1:8b-instruct-q4_K_M",
        "llama3.3:70b-instruct-q4_K_M",
        "llama4:16x17b",
    }


def test_pattern_with_no_catalog_match_falls_back_to_installed_tag():
    result = resolve_custom_models(["my-custom-model:latest"], LLM_MODELS,
                                    installed_tags=["my-custom-model:latest"])
    assert len(result) == 1
    m = result[0]
    assert m["tag"] == "my-custom-model:latest"
    assert m["short"] == "my-custom-model-latest"
    assert "custom" in m["label"]


def test_pattern_not_installed_and_not_in_catalog_matches_nothing():
    result = resolve_custom_models(["nonexistent-model*"], LLM_MODELS, installed_tags=["qwen3.5:4b"])
    assert result == []


def test_wildcard_matching_catalog_does_not_pull_in_unrelated_installed_tags():
    # Installed custom entries still have to match the same pattern.
    result = resolve_custom_models(["llama*"], LLM_MODELS, installed_tags=["qwen3.5:4b"])
    tags = {m["tag"] for m in result}
    assert "qwen3.5:4b" not in tags


def test_wildcard_unions_catalog_matches_with_matching_installed_custom_tags():
    result = resolve_custom_models(
        ["llama*"], LLM_MODELS,
        installed_tags=["llama-local-finetune", "unrelated-custom"],
    )
    tags = {m["tag"] for m in result}
    assert "llama-local-finetune" in tags
    assert "unrelated-custom" not in tags
    assert {m["tag"] for m in LLM_MODELS if m["tag"].startswith("llama")} < tags


def test_custom_wildcard_matches_multiple_installed_tags():
    result = resolve_custom_models(
        ["my-finetune*"], LLM_MODELS,
        installed_tags=["my-finetune:v1", "my-finetune:v2", "other-model"],
    )
    tags = {m["tag"] for m in result}
    assert tags == {"my-finetune:v1", "my-finetune:v2"}


def test_overlapping_custom_patterns_do_not_duplicate_installed_tag():
    result = resolve_custom_models(
        ["my-finetune*", "*:v1"], LLM_MODELS,
        installed_tags=["my-finetune:v1"],
    )
    assert [m["tag"] for m in result] == ["my-finetune:v1"]


def test_mixed_catalog_and_custom_patterns():
    result = resolve_custom_models(
        ["phi4-mini", "qwen3.5:4b"], LLM_MODELS, installed_tags=["qwen3.5:4b"],
    )
    tags = {m["tag"] for m in result}
    assert tags == {"phi4-mini", "qwen3.5:4b"}


def test_sanitize_tag_to_short_replaces_colons_and_slashes():
    assert sanitize_tag_to_short("qwen3.5:4b-instruct") == "qwen3.5-4b-instruct"
    assert sanitize_tag_to_short("someorg/some-model:latest") == "someorg-some-model-latest"
