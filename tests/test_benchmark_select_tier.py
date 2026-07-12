from benchmark import select_tier
from models import (
    IMAGE_MODELS,
    LLM_MODELS,
    LLM_MODELS_XSMALL,
    LLM_MODELS_SMALL,
    LLM_MODELS_MEDIUM,
    LLM_MODELS_LARGE,
)


def test_no_maxtier_returns_everything():
    llm_models, tier_label, image_models = select_tier(None, IMAGE_MODELS)
    assert llm_models == LLM_MODELS
    assert image_models == IMAGE_MODELS


def test_xsmall_tier_only_includes_xsmall():
    llm_models, tier_label, image_models = select_tier("xsmall", IMAGE_MODELS)
    assert llm_models == LLM_MODELS_XSMALL
    assert all(m["tier"] == "xsmall" for m in image_models)


def test_small_tier_is_cumulative():
    llm_models, tier_label, image_models = select_tier("small", IMAGE_MODELS)
    assert llm_models == LLM_MODELS_XSMALL + LLM_MODELS_SMALL
    assert all(m["tier"] in ("xsmall", "small") for m in image_models)


def test_medium_tier_is_cumulative():
    llm_models, tier_label, image_models = select_tier("medium", IMAGE_MODELS)
    assert llm_models == LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM
    assert all(m["tier"] in ("xsmall", "small", "medium") for m in image_models)


def test_large_tier_includes_everything_same_as_no_cap():
    llm_models, tier_label, image_models = select_tier("large", IMAGE_MODELS)
    assert llm_models == LLM_MODELS
    assert image_models == IMAGE_MODELS


def test_medium_tier_excludes_large_tier_image_models():
    _, _, image_models = select_tier("medium", IMAGE_MODELS)
    large_image_shorts = {m["short"] for m in IMAGE_MODELS if m["tier"] == "large"}
    selected_shorts = {m["short"] for m in image_models}
    assert not (large_image_shorts & selected_shorts)


def test_tier_label_is_human_readable_and_distinct_per_tier():
    labels = {t: select_tier(t, IMAGE_MODELS)[1] for t in ("xsmall", "small", "medium", "large")}
    assert len(set(labels.values())) == 4
