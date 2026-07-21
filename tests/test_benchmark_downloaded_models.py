from benchmark import downloaded_models, resolve_model_scopes
from models import LLM_MODELS

CATALOG = [
    {"tag": "phi4-mini", "label": "Phi 4 Mini"},
    {"tag": "llama3.2:3b-instruct-q4_K_M", "label": "Llama 3.2 3B"},
    {"tag": "nemotron-3-super:120b", "label": "Nemotron 3 Super 120B"},
]


def test_returns_only_downloaded_catalog_entries():
    result = downloaded_models(CATALOG, ["phi4-mini"])
    assert [m["tag"] for m in result] == ["phi4-mini"]


def test_preserves_catalog_order_not_installed_tags_order():
    result = downloaded_models(CATALOG, ["nemotron-3-super:120b", "phi4-mini"])
    assert [m["tag"] for m in result] == ["phi4-mini", "nemotron-3-super:120b"]


def test_no_downloaded_tags_returns_empty_list():
    assert downloaded_models(CATALOG, []) == []


def test_installed_tag_not_in_catalog_is_ignored():
    result = downloaded_models(CATALOG, ["phi4-mini", "some-custom-model"])
    assert [m["tag"] for m in result] == ["phi4-mini"]


def test_every_catalog_model_downloaded():
    result = downloaded_models(CATALOG, [m["tag"] for m in CATALOG])
    assert result == CATALOG


def test_resolve_model_scopes_uses_each_engines_installed_inventory():
    first_tag = LLM_MODELS[0]["tag"]
    second_tag = LLM_MODELS[1]["tag"]

    _, first_concurrency = resolve_model_scopes(
        LLM_MODELS, [first_tag], patterns=None, concurrency_enabled=True,
    )
    _, second_concurrency = resolve_model_scopes(
        LLM_MODELS, [second_tag], patterns=None, concurrency_enabled=True,
    )

    assert [m["tag"] for m in first_concurrency] == [first_tag]
    assert [m["tag"] for m in second_concurrency] == [second_tag]


def test_resolve_model_scopes_skips_concurrency_selection_when_disabled():
    run_models, concurrency_models = resolve_model_scopes(
        CATALOG, [CATALOG[0]["tag"]], patterns=None, concurrency_enabled=False,
    )
    assert run_models == CATALOG
    assert concurrency_models == []
