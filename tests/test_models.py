from scripts.workloads.models import (
    EMBED_MODELS,
    IMAGE_MODELS,
    LLM_MODELS,
    LLM_MODELS_XSMALL,
    LLM_MODELS_SMALL,
    LLM_MODELS_MEDIUM,
    LLM_MODELS_LARGE,
    image_checkpoint_folder,
    image_checkpoint_groups,
    image_checkpoint_loader,
    image_checkpoint_path,
)

ALL_LLM_TIERS = [LLM_MODELS_XSMALL, LLM_MODELS_SMALL, LLM_MODELS_MEDIUM, LLM_MODELS_LARGE]


def test_llm_models_is_concatenation_of_tiers():
    assert LLM_MODELS == LLM_MODELS_XSMALL + LLM_MODELS_SMALL + LLM_MODELS_MEDIUM + LLM_MODELS_LARGE


def test_xsmall_and_small_rosters_preserve_the_worker_model_structure():
    assert [model["short"] for model in LLM_MODELS_XSMALL] == [
        "gemma3-1b", "granite4.1-3b-q4", "qwen3.5-4b-q4",
    ]
    assert [model["short"] for model in LLM_MODELS_SMALL] == [
        "granite4.1-8b-q4", "qwen3.5-9b-q4", "gemma4-12b-q4",
    ]


def test_granite_and_qwen_worker_models_form_cross_tier_scaling_pairs():
    pairs = {
        "granite": ("granite4.1-3b-q4", "granite4.1-8b-q4"),
        "qwen": ("qwen3.5-4b-q4", "qwen3.5-9b-q4"),
    }
    for xsmall_short, small_short in pairs.values():
        assert any(model["short"] == xsmall_short for model in LLM_MODELS_XSMALL)
        assert any(model["short"] == small_short for model in LLM_MODELS_SMALL)


def test_gemma4_12b_uses_bartowski_q4_model():
    model = next(model for model in LLM_MODELS_SMALL
                 if model["short"] == "gemma4-12b-q4")
    assert model["hf_repo"] == "bartowski/gemma-4-12B-it-GGUF"
    assert model["hf_file"] == "gemma-4-12B-it-Q4_K_M.gguf"
    assert model["download_size"] == "~7.7 GB"


def test_medium_roster_preserves_dense_and_sparse_architecture_mix():
    assert [model["short"] for model in LLM_MODELS_MEDIUM] == [
        "gemma4-26b-a4b-q4",
        "qwen3.8-27b-q4",
        "nemotron3.5-lightning-30b-a3b",
    ]


def test_medium_models_use_qualified_q4_and_vllm_artifacts():
    models = {model["short"]: model for model in LLM_MODELS_MEDIUM}
    assert models["gemma4-26b-a4b-q4"] == {
        "tag": "gemma4:26b-a4b-it-ud-q4_K_M",
        "label": "Gemma 4 26B-A4B 4-Bit Quantization",
        "short": "gemma4-26b-a4b-q4", "tier": "medium",
        "download_size": "~16.9 GB", "params_b": 26,
        "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "hf_file": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "vllm_repo": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "vllm_download_size": "~17.2 GB", "vllm_tool_parser": "gemma4",
    }
    assert models["qwen3.8-27b-q4"]["hf_file"] == "Qwen3.8-27B-UD-Q4_K_M.gguf"
    assert models["qwen3.8-27b-q4"]["vllm_repo"] == "cyankiwi/Qwen3.8-27B-AWQ-INT4"
    assert models["nemotron3.5-lightning-30b-a3b"]["hf_file"] == \
        "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf"
    assert models["nemotron3.5-lightning-30b-a3b"]["vllm_repo"] == \
        "Local-Axiom-AI/Nemotron-3.5-Lightning-awq"


def test_large_roster_preserves_distinct_baseline_agent_and_planner_roles():
    assert [model["short"] for model in LLM_MODELS_LARGE] == [
        "llama3.3-70b-q4",
        "qwen3-coder-next-80b-a3b-q4",
        "nemotron3-super-120b",
    ]


def test_qwen3_coder_next_uses_complete_official_q4_multipart_set():
    model = next(model for model in LLM_MODELS_LARGE
                 if model["short"] == "qwen3-coder-next-80b-a3b-q4")
    assert model["hf_repo"] == "Qwen/Qwen3-Coder-Next-GGUF"
    assert model["hf_file"] == [
        f"Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-{part:05d}-of-00004.gguf"
        for part in range(1, 5)
    ]


def test_each_llm_tier_sorted_by_params():
    for tier in ALL_LLM_TIERS:
        params = [m["params_b"] for m in tier]
        assert params == sorted(params)


def test_llm_tags_and_shorts_unique():
    tags = [m["tag"] for m in LLM_MODELS]
    shorts = [m["short"] for m in LLM_MODELS]
    assert len(tags) == len(set(tags))
    assert len(shorts) == len(set(shorts))


def test_llm_models_have_required_keys():
    required = {"tag", "label", "short", "tier", "download_size", "params_b", "hf_repo", "hf_file"}
    for m in LLM_MODELS:
        assert required <= m.keys()


def test_llm_models_tier_matches_source_list():
    expected = {
        "xsmall": LLM_MODELS_XSMALL,
        "small":  LLM_MODELS_SMALL,
        "medium": LLM_MODELS_MEDIUM,
        "large":  LLM_MODELS_LARGE,
    }
    for tier_name, models in expected.items():
        for m in models:
            assert m["tier"] == tier_name


def test_embed_models_have_required_keys():
    required = {"tag", "label", "short", "download_size", "hf_repo", "hf_file"}
    for m in EMBED_MODELS:
        assert required <= m.keys()


def test_hf_file_is_string_or_list_of_strings():
    for m in LLM_MODELS + EMBED_MODELS:
        hf_file = m["hf_file"]
        if isinstance(hf_file, list):
            assert hf_file and all(isinstance(f, str) for f in hf_file)
        else:
            assert isinstance(hf_file, str)


def test_image_models_shorts_unique():
    shorts = [m["short"] for m in IMAGE_MODELS]
    assert len(shorts) == len(set(shorts))


def test_image_models_valid_tier():
    valid_tiers = {"xsmall", "small", "medium", "large"}
    for m in IMAGE_MODELS:
        assert m["tier"] in valid_tiers


def test_image_models_have_required_keys():
    required = {"label", "checkpoint", "workflow", "steps", "cfg", "sampler", "scheduler", "short", "tier"}
    for m in IMAGE_MODELS:
        assert required <= m.keys()


def test_z_image_catalog_uses_complete_official_comfyui_pipeline(tmp_path):
    model = next(model for model in IMAGE_MODELS if model["short"] == "z-image-turbo")
    assert model["checkpoint"] == "z_image_turbo_bf16.safetensors"
    assert image_checkpoint_folder(model) == "diffusion_models"
    assert image_checkpoint_loader(model) == "UNETLoader"
    assert image_checkpoint_path(model, tmp_path) == \
        tmp_path / "diffusion_models" / "z_image_turbo_bf16.safetensors"
    assert model["checkpoint_repo"] == "Comfy-Org/z_image_turbo"
    assert model["checkpoint_remote"] == \
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors"
    assert [(asset["folder"], asset["name"], asset["remote"])
            for asset in model["support_assets"]] == [
        ("text_encoders", "qwen_3_4b.safetensors",
         "split_files/text_encoders/qwen_3_4b.safetensors"),
        ("vae", "ae.safetensors", "split_files/vae/ae.safetensors"),
    ]


def test_standard_image_checkpoint_defaults_remain_stable(tmp_path):
    model = next(model for model in IMAGE_MODELS if model["short"] == "sdxl")
    assert image_checkpoint_folder(model) == "checkpoints"
    assert image_checkpoint_loader(model) == "CheckpointLoaderSimple"
    assert image_checkpoint_path(model, tmp_path) == \
        tmp_path / "checkpoints" / model["checkpoint"]


def test_image_checkpoint_groups_separate_comfyui_loader_inventories():
    groups = image_checkpoint_groups(IMAGE_MODELS)
    assert groups["UNETLoader"] == {"z_image_turbo_bf16.safetensors"}
    assert groups["CheckpointLoaderSimple"] == {
        "v1-5-pruned-emaonly.safetensors", "sd_xl_base_1.0.safetensors",
        "flux1-dev.safetensors", "flux2-dev.safetensors",
    }


def test_gated_image_models_expose_their_hugging_face_license_pages():
    urls = {model["short"]: model.get("license_url") for model in IMAGE_MODELS}
    assert urls == {
        "sd15": None,
        "sdxl": None,
        "z-image-turbo": None,
        "flux-dev": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
        "flux2-dev": "https://huggingface.co/black-forest-labs/FLUX.2-dev",
    }
