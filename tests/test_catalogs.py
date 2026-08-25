from scripts.results.catalogs import CATALOG_VERSION, HARDWARE_CATALOG, catalog_bundle, model_catalog, recommendation_eligible
from scripts.workloads.models import EMBED_MODELS, IMAGE_MODELS, LLM_MODELS
from scripts.workloads.model_variants import expanded_variant_catalog


def test_catalog_has_stable_unique_hardware_identities_and_sources():
    ids = [item["id"] for item in HARDWARE_CATALOG]
    assert len(ids) == len(set(ids))
    assert all(item["source"].startswith("https://") for item in HARDWARE_CATALOG)
    assert all(item["qualification"] == "unqualified" for item in HARDWARE_CATALOG)


def test_model_catalog_covers_benchmark_catalog_with_stable_ids():
    records = model_catalog()
    expected = len(expanded_variant_catalog(LLM_MODELS)) + len(EMBED_MODELS) + len(IMAGE_MODELS)
    assert len(records) == expected
    assert len({record["id"] for record in records}) == expected
    assert all(record["workloads"] and record["runtimes"] for record in records)
    assert all(record["distribution"] == "download_by_reference" for record in records)


def test_download_by_reference_does_not_block_supported_recommendation():
    record = model_catalog()[0]
    assert recommendation_eligible(record) is True
    record["distribution"] = "bundled"
    assert recommendation_eligible(record) is False
    record["license"] = {"status": "verified", "identifier": "example"}
    assert recommendation_eligible(record) is True


def test_catalog_bundle_is_versioned_and_returns_independent_data():
    first, second = catalog_bundle(), catalog_bundle()
    assert first["version"] == CATALOG_VERSION
    first["hardware"][0]["product"] = "changed"
    assert second["hardware"][0]["product"] != "changed"


def test_quantization_metadata_preserves_known_and_unknown_values():
    records = {record["id"]: record for record in model_catalog()}
    assert records["llm:gemma3:1b-it-q4_K_M"]["quantization"] == "q4_k_m"
    assert records["llm:gemma3:1b-it-q6_K"]["quantization"] == "q6_k"
    assert records["llm:gemma3:1b-it-q8_0"]["quantization"] == "q8_0"
    assert records["image:v1-5-pruned-emaonly.safetensors"]["quantization"] is None
