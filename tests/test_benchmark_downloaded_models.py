from benchmark import downloaded_models

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
