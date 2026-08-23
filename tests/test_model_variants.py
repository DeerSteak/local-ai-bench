import pytest

from scripts.workloads.model_variants import (
    default_model_variant, expanded_model_variants, validate_model_variants,
)


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
    assert "variants" not in expanded[0]
    assert default_model_variant(model())["tag"] == "demo:q4_K_M"


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
