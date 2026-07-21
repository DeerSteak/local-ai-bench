from models import (
    EMBED_MODELS,
    IMAGE_MODELS,
    LLM_MODELS,
    LLM_MODELS_XSMALL,
    LLM_MODELS_SMALL,
    LLM_MODELS_MEDIUM,
    LLM_MODELS_LARGE,
)

ALL_LLM_TIERS = [LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE]


def test_llm_models_is_concatenation_of_tiers():
    assert LLM_MODELS == LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE


def test_each_llm_tier_sorted_by_params():
    for tier in ALL_LLM_TIERS:
        params = [m["params_b"] for m in tier]
        assert params == sorted(params)


def test_llm_tags_and_shorts_unique():
    tags = [m["tag"] for m in LLM_MODELS]
    shorts = [m["short"] for m in LLM_MODELS]
    assert len(tags) == len(set(tags))
    assert len(shorts) == len(set(shorts))


def test_llm_models_have_required_keys():
    required = {"tag", "label", "short", "tier", "download_size", "params_b", "hf_repo", "hf_file"}
    for m in LLM_MODELS:
        assert required <= m.keys()


def test_llm_models_tier_matches_source_list():
    expected = {
        "xsmall": LLM_MODELS_XSMALL,
        "small":  LLM_MODELS_SMALL,
        "medium": LLM_MODELS_MEDIUM,
        "large":  LLM_MODELS_LARGE,
    }
    for tier_name, models in expected.items():
        for m in models:
            assert m["tier"] == tier_name


def test_embed_models_have_required_keys():
    required = {"tag", "label", "short", "download_size", "hf_repo", "hf_file"}
    for m in EMBED_MODELS:
        assert required <= m.keys()


def test_hf_file_is_string_or_list_of_strings():
    for m in LLM_MODELS + EMBED_MODELS:
        hf_file = m["hf_file"]
        if isinstance(hf_file, list):
            assert hf_file and all(isinstance(f, str) for f in hf_file)
        else:
            assert isinstance(hf_file, str)


def test_image_models_shorts_unique():
    shorts = [m["short"] for m in IMAGE_MODELS]
    assert len(shorts) == len(set(shorts))


def test_image_models_valid_tier():
    valid_tiers = {"xsmall", "small", "medium", "large"}
    for m in IMAGE_MODELS:
        assert m["tier"] in valid_tiers


def test_image_models_have_required_keys():
    required = {"label", "checkpoint", "workflow", "steps", "cfg", "sampler", "scheduler", "short", "tier"}
    for m in IMAGE_MODELS:
        assert required <= m.keys()
