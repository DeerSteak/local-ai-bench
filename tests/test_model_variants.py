import pytest

from scripts.workloads.model_variants import (
    default_model_variant, expanded_model_variants, expanded_variant_catalog,
    normalize_variant_selectors,
    select_model_variants, validate_model_variants,
)
from scripts.workloads.models import LLM_MODELS


def model(**overrides):
    value = {
        "tag": "demo:q4_K_M", "short": "demo-q4", "label": "Demo",
        "base_model": "demo", "hf_repo": "owner/demo", "hf_file": "demo-q4.gguf",
        "download_size": "~4 GB", "params_b": 7,
        "variants": [
            {
                "tag": "demo:q4_K_M", "short": "demo-q4", "quantization": "Q4_K_M",
                "hf_repo": "owner/demo", "hf_file": "demo-q4.gguf",
                "download_size": "~4 GB", "default": True,
            },
            {
                "tag": "demo:q8_0", "short": "demo-q8", "quantization": "Q8_0",
                "hf_repo": "owner/demo", "hf_file": ["demo-q8-1.gguf", "demo-q8-2.gguf"],
                "download_size": "~8 GB",
            },
        ],
    }
    value.update(overrides)
    return value


def test_single_variant_model_is_unchanged_and_copied():
    original = {"tag": "legacy:q4", "short": "legacy", "hf_repo": "owner/legacy"}

    expanded = expanded_model_variants(original)

    assert expanded == [original]
    assert expanded[0] is not original
    assert default_model_variant(original) == original


def test_multi_variant_model_expands_distinct_executable_records():
    expanded = expanded_model_variants(model())

    assert [(item["tag"], item["base_model"], item["variant"]) for item in expanded] == [
        ("demo:q4_K_M", "demo", "Q4_K_M"),
        ("demo:q8_0", "demo", "Q8_0"),
    ]
    assert expanded[1]["hf_file"] == ["demo-q8-1.gguf", "demo-q8-2.gguf"]
    assert expanded[1]["label"] == "Demo — Q8_0 (~8 GB)"
    assert "variants" not in expanded[0]
    assert default_model_variant(model())["tag"] == "demo:q4_K_M"


def test_catalog_variants_are_valid_and_default_preserves_legacy_gemma_identity():
    for catalog_model in LLM_MODELS:
        validate_model_variants(catalog_model)

    gemma = next(item for item in LLM_MODELS if item.get("base_model") == "gemma3:1b-it")
    assert [item["variant"] for item in expanded_model_variants(gemma)] == [
        "Q4_K_M", "Q6_K", "Q8_0",
    ]
    assert default_model_variant(gemma)["tag"] == "gemma3:1b-it-q4_K_M"
    assert default_model_variant(gemma)["short"] == "gemma3-1b"
    assert {item["tag"] for item in expanded_variant_catalog([gemma])} == {
        "gemma3:1b-it-q4_K_M", "gemma3:1b-it-q6_K", "gemma3:1b-it-q8_0",
    }


def test_variant_selectors_preserve_catalog_order_and_unselected_models():
    catalog = [model(), {"tag": "other:q4", "short": "other"}]
    selections = normalize_variant_selectors(
        ["demo=Q8_0", "demo=Q4_K_M"], catalog,
    )

    selected = select_model_variants(catalog, selections)

    assert [item["tag"] for item in selected] == ["demo:q8_0", "demo:q4_K_M", "other:q4"]
    assert selected[0]["base_model"] == "demo"
    assert selected[0]["variant"] == "Q8_0"


@pytest.mark.parametrize("selectors, message", [
    ([], "must not be empty"),
    (["demo"], "BASE=VARIANT"),
    (["demo="], "non-empty"),
    (["missing=Q4_K_M"], "unknown model/variant pair"),
    (["demo=Q5_K"], "unknown model/variant pair"),
    (["demo=Q4_K_M", "demo=Q4_K_M"], "duplicate model/variant pair"),
])
def test_invalid_variant_selectors_are_rejected(selectors, message):
    with pytest.raises(ValueError, match=message):
        normalize_variant_selectors(selectors, [model()])


def test_variant_selection_rejects_base_outside_selected_model_scope():
    with pytest.raises(ValueError, match="base model is not selected"):
        select_model_variants(
            [{"tag": "other:q4", "short": "other"}], {"demo": ("Q4_K_M",)},
        )


@pytest.mark.parametrize("mutate, message", [
    (lambda value: value.pop("base_model"), "requires base_model"),
    (lambda value: value.update(variants=[]), "at least two"),
    (lambda value: value["variants"][0].pop("hf_repo"), "missing fields"),
    (lambda value: value["variants"][1].update(short="demo-q4"), "duplicate short"),
    (lambda value: value["variants"][1].update(quantization="Q4_K_M"),
     "duplicate quantization"),
    (lambda value: value["variants"][0].pop("default"), "exactly one default"),
    (lambda value: value["variants"][1].update(default=True), "exactly one default"),
    (lambda value: value["variants"][0].update(hf_file="wrong.gguf"),
     "match top-level hf_file"),
])
def test_malformed_variant_catalog_is_rejected(mutate, message):
    value = model()
    mutate(value)

    with pytest.raises(ValueError, match=message):
        validate_model_variants(value)
