from scripts.runtime import config


def test_context_lengths_ascending():
    assert config.CONTEXT_LENGTHS == sorted(config.CONTEXT_LENGTHS)
    assert len(config.CONTEXT_LENGTHS) == len(set(config.CONTEXT_LENGTHS))


def test_results_dir_under_script_dir():
    assert config.RESULTS_DIR.parent == config.SCRIPT_DIR


def test_comfyui_dir_under_script_dir():
    assert config.COMFYUI_DIR == config.SCRIPT_DIR / "ComfyUI"


def test_comfyui_models_are_managed_separately_from_program():
    assert config.COMFYUI_MODELS_DIR == config.MODELS_DIR / "comfyui"
    assert config.COMFYUI_MODELS_DIR != config.COMFYUI_DIR / "models"


def test_n_runs_positive():
    assert config.N_RUNS >= 1


def test_urls_have_scheme():
    assert config.LLAMACPP_URL.startswith("http://")
    assert config.COMFYUI_URL.startswith("http://")


def test_models_dir_under_script_dir():
    assert config.MODELS_DIR == config.SCRIPT_DIR / "models"
