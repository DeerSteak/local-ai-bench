from scripts.setup.custom_models import custom_model, load_custom_models, save_custom_model


def test_custom_model_registry_round_trips_and_replaces_same_engine_tag(tmp_path):
    path = tmp_path / "custom.json"
    save_custom_model({"engine": "llamacpp", "tag": "z", "label": "Old"}, path)
    save_custom_model({"engine": "vllm", "tag": "z", "label": "vLLM"}, path)
    save_custom_model({"engine": "llamacpp", "tag": "z", "label": "New"}, path)

    assert load_custom_models(path) == [
        {"engine": "llamacpp", "tag": "z", "label": "New"},
        {"engine": "vllm", "tag": "z", "label": "vLLM"},
    ]
    found = custom_model("llamacpp", "z", path)
    assert found is not None and found["label"] == "New"


def test_custom_model_registry_tolerates_missing_and_invalid_files(tmp_path):
    path = tmp_path / "custom.json"
    assert load_custom_models(path) == []
    path.write_text("not json", encoding="utf-8")
    assert load_custom_models(path) == []
