from scripts.model_identity import model_tag_slug


def test_model_tag_slug_normalizes_registry_and_repository_separators():
    assert model_tag_slug("org/model:Q4_K_M") == "org_model_Q4_K_M"
