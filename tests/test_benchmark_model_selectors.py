import argparse

from benchmark import (
    add_model_selection_arguments,
    filter_models_by_pattern,
    resolve_catalog_scopes,
    resolve_engine_scopes,
    select_tier,
    validate_catalog_scopes,
    validate_engine_scopes,
)
from models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


class FakeEngine:
    def __init__(self, name, installed):
        self.name = name
        self.installed = installed
        self.list_calls = 0

    def list_installed_models(self):
        self.list_calls += 1
        return [{"tag": tag, "size": 1} for tag in self.installed]


def parser():
    result = argparse.ArgumentParser()
    add_model_selection_arguments(result)
    return result


def test_llm_models_is_canonical_selector():
    args = parser().parse_args(["--llm-models", "phi4-mini", "llama*"])
    assert args.llm_models == ["phi4-mini", "llama*"]


def test_models_alias_uses_same_destination():
    args = parser().parse_args(["--models", "phi4-mini", "llama*"])
    assert args.llm_models == ["phi4-mini", "llama*"]


def test_embedding_and_image_selectors_parse_independently():
    args = parser().parse_args([
        "--embedding-models", "nomic*",
        "--image-models", "sd*", "flux-dev",
    ])
    assert args.embedding_models == ["nomic*"]
    assert args.image_models == ["sd*", "flux-dev"]


def test_every_selector_defaults_to_none():
    args = parser().parse_args([])
    assert args.llm_models is None
    assert args.embedding_models is None
    assert args.image_models is None


def test_filter_models_can_match_image_short_instead_of_tag():
    result = filter_models_by_pattern(IMAGE_MODELS, ["flux*"], key="short")
    assert [model["short"] for model in result] == ["flux-dev", "flux2-dev"]


def test_image_short_matching_is_case_sensitive():
    assert filter_models_by_pattern(IMAGE_MODELS, ["FLUX*"], key="short") == []


def test_catalog_scopes_preserve_defaults_when_selectors_omitted():
    embedding_models, image_models = resolve_catalog_scopes(IMAGE_MODELS, None, None)
    assert embedding_models == EMBED_MODELS
    assert image_models == IMAGE_MODELS


def test_catalog_scopes_filter_embedding_tags_and_image_shorts():
    embedding_models, image_models = resolve_catalog_scopes(
        IMAGE_MODELS, ["nomic*"], ["sd*"],
    )
    assert [model["tag"] for model in embedding_models] == ["nomic-embed-text"]
    assert [model["short"] for model in image_models] == ["sd15", "sdxl", "sd35-large"]


def test_catalog_validation_rejects_relevant_empty_embedding_selection():
    errors = validate_catalog_scopes(
        ["emb"], ["missing*"], None, embedding_models=[], image_models=IMAGE_MODELS,
    )
    assert errors == ["--embedding-models missing* matched no embedding models"]


def test_catalog_validation_rejects_relevant_empty_image_selection():
    errors = validate_catalog_scopes(
        ["img"], None, ["missing*"], embedding_models=EMBED_MODELS, image_models=[],
    )
    assert errors == ["--image-models missing* matched no image models"]


def test_catalog_validation_ignores_selectors_for_unselected_workloads():
    errors = validate_catalog_scopes(
        ["llm"], ["missing*"], ["missing*"], embedding_models=[], image_models=[],
    )
    assert errors == []


def test_catalog_validation_accepts_any_nonempty_match():
    errors = validate_catalog_scopes(
        ["emb", "img"], ["nomic*", "missing*"], ["sd15", "missing*"],
        embedding_models=[EMBED_MODELS[0]], image_models=[IMAGE_MODELS[0]],
    )
    assert errors == []


def test_image_selector_narrows_after_maxtier():
    _, _, small_images = select_tier("small", IMAGE_MODELS)
    _, image_models = resolve_catalog_scopes(small_images, None, ["flux*"])
    errors = validate_catalog_scopes(
        ["img"], None, ["flux*"], embedding_models=EMBED_MODELS, image_models=image_models,
    )
    assert image_models == []
    assert errors == ["--image-models flux* matched no image models"]


def test_engine_validation_rejects_empty_normal_llm_scope():
    errors = validate_engine_scopes(
        ["llm"], "fake", ["missing*"], [], [], "small and below",
    )
    assert errors == [
        "--llm-models missing* matched no LLM models in the selected tier "
        "(small and below) or installed for fake"
    ]


def test_engine_validation_rejects_empty_concurrency_scope_separately():
    errors = validate_engine_scopes(
        ["llm", "conc_tool"], "fake", ["phi4-mini"], [LLM_MODELS[0]], [], "all",
    )
    assert errors == [
        "--llm-models phi4-mini matched no downloaded concurrency models for fake"
    ]


def test_engine_validation_ignores_irrelevant_llm_selector():
    errors = validate_engine_scopes(
        ["img"], "fake", ["missing*"], [], [], "all",
    )
    assert errors == []


def test_engine_prepass_does_not_read_inventory_for_catalog_only_normal_run():
    engine = FakeEngine("fake", [])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all",
        ["qwen3.5:4b-q4_K_M"], ["llm"],
    )
    assert engine.list_calls == 0
    assert [model["tag"] for model in scopes[0]["llm_models"]] == [
        "qwen3.5:4b-q4_K_M",
    ]
    assert errors == []


def test_engine_prepass_reads_inventory_when_wildcard_can_match_custom_models():
    engine = FakeEngine("fake", ["llama-local-finetune"])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all", ["llama*"], ["llm"],
    )
    tags = {model["tag"] for model in scopes[0]["llm_models"]}
    assert engine.list_calls == 1
    assert "llama-local-finetune" in tags
    assert {model["tag"] for model in LLM_MODELS if model["tag"].startswith("llama")} < tags
    assert errors == []


def test_engine_prepass_reads_inventory_for_custom_selector():
    engine = FakeEngine("fake", ["my-custom-model"])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all", ["my-custom-model"], ["conv"],
    )
    assert engine.list_calls == 1
    assert [model["tag"] for model in scopes[0]["llm_models"]] == ["my-custom-model"]
    assert errors == []


def test_engine_prepass_ignores_custom_selector_when_no_llm_test_selected():
    engine = FakeEngine("fake", ["my-custom-model"])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all", ["my-custom-model"], ["img"],
    )
    assert engine.list_calls == 0
    assert scopes[0]["llm_models"] == []
    assert errors == []


def test_engine_prepass_reads_inventory_for_concurrency_without_selector():
    installed_tag = LLM_MODELS[-1]["tag"]
    engine = FakeEngine("fake", [installed_tag])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS[:1], "xsmall", None, ["conc_chat"],
    )
    assert engine.list_calls == 1
    assert [model["tag"] for model in scopes[0]["concurrency_models"]] == [installed_tag]
    assert errors == []


def test_engine_prepass_applies_explicit_selector_to_downloaded_concurrency_scope():
    installed_tags = [LLM_MODELS[0]["tag"], LLM_MODELS[1]["tag"]]
    engine = FakeEngine("fake", installed_tags)
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all",
        [installed_tags[1]], ["conc_tool"],
    )
    assert [model["tag"] for model in scopes[0]["concurrency_models"]] == [installed_tags[1]]
    assert errors == []


def test_engine_prepass_aggregates_failure_from_any_engine():
    engines = {
        "first": FakeEngine("first", ["my-custom-model"]),
        "second": FakeEngine("second", []),
    }
    scopes, errors = resolve_engine_scopes(
        ["first", "second"], engines.get, LLM_MODELS, "all",
        ["my-custom-model"], ["llm"],
    )
    assert [scope["engine"] for scope in scopes] == [engines["first"], engines["second"]]
    assert errors == [
        "--llm-models my-custom-model matched no LLM models in the selected tier "
        "(all) or installed for second"
    ]


def test_out_of_tier_catalog_model_does_not_reappear_as_custom():
    selected_tier = LLM_MODELS[:1]
    out_of_tier = LLM_MODELS[-1]["tag"]
    engine = FakeEngine("fake", [out_of_tier])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, selected_tier, "xsmall", [out_of_tier], ["llm"],
    )
    assert scopes[0]["llm_models"] == []
    assert errors == [
        f"--llm-models {out_of_tier} matched no LLM models in the selected tier "
        "(xsmall) or installed for fake"
    ]


def test_installed_embedding_tag_does_not_reappear_as_custom_llm():
    embedding_tag = EMBED_MODELS[0]["tag"]
    engine = FakeEngine("fake", [embedding_tag])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all", [embedding_tag], ["llm"],
    )
    assert engine.list_calls == 0
    assert scopes[0]["llm_models"] == []
    assert errors == [
        f"--llm-models {embedding_tag} matched no LLM models in the selected tier "
        "(all) or installed for fake"
    ]
