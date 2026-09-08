import argparse

import pytest

from scripts.app.benchmark import (
    LLM_TESTS,
    add_model_selection_arguments,
    apply_variant_selections,
    engine_scope_tests,
    filter_models_by_pattern,
    resolve_catalog_scopes,
    resolve_engine_scopes,
    select_tier,
    selected_plan_models,
    validate_catalog_scopes,
    validate_engine_scopes,
)
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS


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


def test_model_variant_is_repeatable_and_model_qualified():
    args = parser().parse_args([
        "--model-variant", "gemma3:1b-it=Q4_K_M",
        "--model-variant", "gemma3:1b-it=Q8_0",
    ])
    assert args.model_variants == ["gemma3:1b-it=Q4_K_M", "gemma3:1b-it=Q8_0"]


def test_variant_selection_expands_llm_and_downloaded_concurrency_scopes():
    gemma = LLM_MODELS[0]
    scope = {
        "name": "llamacpp", "llm_models": [gemma], "concurrency_models": [gemma],
    }

    apply_variant_selections(
        [scope], ["gemma3:1b-it=Q4_K_M", "gemma3:1b-it=Q8_0"],
        ["llamacpp"], ["llm", "conc_chat"],
    )

    assert [model["variant"] for model in scope["llm_models"]] == ["Q4_K_M", "Q8_0"]
    assert [model["variant"] for model in scope["concurrency_models"]] == [
        "Q4_K_M", "Q8_0",
    ]


def test_variant_selection_can_compare_native_and_vulkan_llamacpp():
    gemma = LLM_MODELS[0]
    scopes = [
        {"name": name, "llm_models": [gemma], "concurrency_models": []}
        for name in ("llamacpp", "llamacpp-vulkan")
    ]
    apply_variant_selections(
        scopes, ["gemma3:1b-it=Q8_0"],
        ["llamacpp", "llamacpp-vulkan"], ["llm"],
    )
    assert all([model["variant"] for model in scope["llm_models"]] == ["Q8_0"]
               for scope in scopes)


def test_ordinary_catalog_selection_emits_complete_default_variant_identity():
    scope = {
        "name": "llamacpp", "llm_models": [LLM_MODELS[0]],
        "concurrency_models": [],
    }

    apply_variant_selections([scope], None, ["llamacpp"], ["llm"])
    identities = selected_plan_models(
        ["llm"], scope["llm_models"], [], [], [],
    )["llm"]

    assert identities == [{
        "tag": "gemma3:1b-it-q4_K_M", "short": "gemma3-1b", "params_b": 1,
        "base_model": "gemma3:1b-it", "variant": "Q4_K_M",
    }]


def test_variant_selection_rejects_vllm_before_execution():
    with pytest.raises(ValueError, match="requires only llama.cpp engines"):
        apply_variant_selections(
            [], ["gemma3:1b-it=Q4_K_M"], ["vllm"], ["llm"],
        )


def test_variant_selection_rejects_explicit_variants_that_are_not_installed():
    gemma = LLM_MODELS[0]
    scope = {
        "name": "llamacpp",
        "inventory_loaded": True,
        "installed_tags": frozenset({"gemma3:1b-it-q4_K_M"}),
        "llm_models": [gemma],
        "concurrency_models": [],
    }

    with pytest.raises(ValueError, match=(
        "selected model variants are not installed: gemma3:1b-it-q8_0"
    )):
        apply_variant_selections(
            [scope], ["gemma3:1b-it=Q4_K_M", "gemma3:1b-it=Q8_0"],
            ["llamacpp"], ["llm"],
        )


def test_variant_selection_checks_only_explicit_families_for_installation():
    gemma, granite = LLM_MODELS[:2]
    scope = {
        "name": "llamacpp",
        "inventory_loaded": True,
        "installed_tags": frozenset({"gemma3:1b-it-q6_K"}),
        "llm_models": [gemma, granite],
        "concurrency_models": [],
    }

    apply_variant_selections(
        [scope], ["gemma3:1b-it=Q6_K"], ["llamacpp"], ["llm"],
    )

    assert [model["tag"] for model in scope["llm_models"]] == [
        "gemma3:1b-it-q6_K", "granite4.1:3b-q4_K_M",
    ]


def test_every_selector_defaults_to_none():
    args = parser().parse_args([])
    assert args.llm_models is None
    assert args.embedding_models is None
    assert args.image_models is None
    assert args.model_variants is None


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
    assert [model["short"] for model in image_models] == ["sd15", "sdxl"]


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


def test_llamabench_shares_the_llm_test_scope():
    assert "llamabench" in LLM_TESTS
    assert "vllmbench" in LLM_TESTS


def test_engine_validation_rejects_empty_llamabench_scope():
    errors = validate_engine_scopes(
        ["llamabench"], "fake", ["missing*"], [], [], "small and below",
    )
    assert errors == [
        "--llm-models missing* matched no LLM models in the selected tier "
        "(small and below) or installed for fake"
    ]


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


def test_engine_prepass_requires_explicit_catalog_model_to_be_installed():
    engine = FakeEngine("fake", [])
    scopes, errors = resolve_engine_scopes(
        ["fake"], lambda _: engine, LLM_MODELS, "all",
        ["qwen3.5:4b-q4_K_M"], ["llm"],
    )
    assert engine.list_calls == 1
    assert scopes[0]["llm_models"] == []
    assert errors == [
        "--llm-models qwen3.5:4b-q4_K_M matched no LLM models in the selected tier "
        "(all) or installed for fake",
    ]


def test_engine_prepass_reads_inventory_for_explicit_variant_without_llm_selector():
    engine = FakeEngine("llamacpp", ["gemma3:1b-it-q6_K"])
    scopes, errors = resolve_engine_scopes(
        ["llamacpp"], lambda _: engine, [LLM_MODELS[0]], "extra-small",
        None, ["llm"], variant_selectors=["gemma3:1b-it=Q6_K"],
    )

    apply_variant_selections(
        scopes, ["gemma3:1b-it=Q6_K"], ["llamacpp"], ["llm"],
    )

    assert engine.list_calls == 1
    assert [model["tag"] for model in scopes[0]["llm_models"]] == [
        "gemma3:1b-it-q6_K",
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
    assert tags == {"llama-local-finetune"}
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


def test_engine_prepass_allows_a_model_owned_by_only_one_selected_engine():
    engines = {
        "first": FakeEngine("first", ["my-custom-model"]),
        "second": FakeEngine("second", []),
    }
    scopes, errors = resolve_engine_scopes(
        ["first", "second"], engines.get, LLM_MODELS, "all",
        ["my-custom-model"], ["llm"],
    )
    assert [scope["engine"] for scope in scopes] == [engines["first"], engines["second"]]
    assert errors == []
    assert [model["tag"] for model in scopes[0]["llm_models"]] == ["my-custom-model"]
    assert scopes[1]["llm_models"] == []
    assert engine_scope_tests(["llm"], scopes[0], include_images=True) == ["llm"]
    assert engine_scope_tests(["llm"], scopes[1], include_images=True) == []


def test_engine_prepass_partitions_catalog_custom_and_embedding_models_by_engine():
    catalog_tag = LLM_MODELS[0]["tag"]
    embedding_tag = EMBED_MODELS[0]["tag"]
    engines = {
        "llamacpp": FakeEngine("llamacpp", [catalog_tag, "llama-custom"]),
        "vllm": FakeEngine("vllm", ["vllm-custom", embedding_tag]),
    }
    scopes, errors = resolve_engine_scopes(
        ["llamacpp", "vllm"], engines.get, LLM_MODELS, "all",
        [catalog_tag, "*-custom"], ["llm", "emb"],
        embedding_models=EMBED_MODELS, embedding_patterns=[embedding_tag],
    )
    assert errors == []
    assert [[model["tag"] for model in scope["llm_models"]] for scope in scopes] == [
        [catalog_tag, "llama-custom"], ["vllm-custom"],
    ]
    assert [scope["embedding_models"] for scope in scopes] == [[], [EMBED_MODELS[0]]]
    assert engine_scope_tests(["llm", "emb"], scopes[0], include_images=True) == ["llm"]
    assert engine_scope_tests(["llm", "emb"], scopes[1], include_images=True) == [
        "llm", "emb",
    ]


def test_multi_engine_prepass_rejects_selector_missing_from_every_engine():
    engines = {
        "llamacpp": FakeEngine("llamacpp", []),
        "vllm": FakeEngine("vllm", []),
    }
    _scopes, errors = resolve_engine_scopes(
        ["llamacpp", "vllm"], engines.get, LLM_MODELS, "all",
        ["missing-custom"], ["llm"],
    )
    assert errors == [
        "--llm-models missing-custom matched no LLM models in the selected tier "
        "(all) or installed for the selected engines",
    ]


def test_native_selector_is_validated_only_by_its_own_engine():
    engines = {
        "llamacpp": FakeEngine("llamacpp", []),
        "vllm": FakeEngine("vllm", ["my-custom-model"]),
    }
    scopes, errors = resolve_engine_scopes(
        ["llamacpp", "vllm"], engines.get, LLM_MODELS, "all",
        ["my-custom-model"], ["vllmbench"],
    )
    assert [model["tag"] for model in scopes[1]["llm_models"]] == ["my-custom-model"]
    assert errors == []


def test_vllmbench_rejects_an_empty_explicit_model_scope():
    engine = FakeEngine("vllm", [])
    _scopes, errors = resolve_engine_scopes(
        ["vllm"], lambda _: engine, LLM_MODELS, "all",
        ["missing-custom"], ["vllmbench"],
    )
    assert errors == [
        "--llm-models missing-custom matched no LLM models in the selected tier "
        "(all) or installed for vllm"
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
    assert engine.list_calls == 1
    assert scopes[0]["llm_models"] == []
    assert errors == [
        f"--llm-models {embedding_tag} matched no LLM models in the selected tier "
        "(all) or installed for fake"
    ]


def test_plan_models_exclude_families_without_selected_workloads():
    llm = [{"tag": "llm:4b", "short": "llm"}]
    concurrency = [{"tag": "conc:4b", "short": "conc"}]
    embeddings = [{"tag": "embed", "short": "embed"}]
    images = [{"short": "sdxl"}]
    assert selected_plan_models(
        ["llm", "emb"], llm, concurrency, embeddings, images,
    ) == {
        "llm": llm,
        "concurrency": [],
        "embeddings": embeddings,
        "images": [],
    }


def test_plan_models_include_short_only_images_for_selected_image_workload():
    scoped = selected_plan_models(["img"], [], [], [], [{"short": "sdxl"}])
    assert scoped == {
        "llm": [], "concurrency": [], "embeddings": [],
        "images": [{"short": "sdxl"}],
    }


def test_plan_records_models_for_vllmbench():
    llm = [{"tag": "llm:4b", "short": "llm"}]
    scoped = selected_plan_models(["vllmbench"], llm, [], [], [])
    assert scoped["llm"] == llm
