import pytest

from scripts.release.qualification_automation import execution_recipe_gaps
from scripts.release.qualification_recipe import TARGETS, TARGET_ACCELERATORS, build_recipe


@pytest.mark.parametrize("target_id", TARGETS)
def test_every_declared_target_generates_a_complete_recipe(tmp_path, target_id):
    recipe = build_recipe(
        target_id=target_id, root=tmp_path, output=tmp_path / "evidence",
        baseline_version="baseline", target_version="target",
        python_executable="/usr/bin/python3",
        accelerator_identity=TARGET_ACCELERATORS[target_id],
    )
    assert execution_recipe_gaps(recipe) == []
    assert recipe["target"]["id"] == target_id
    assert recipe["target"]["runtime_version"] == "target"
    assert "--smoke-output" in recipe["steps"]["upgrade"]["command"]
    assert recipe["steps"]["cancellation"]["interrupt_when_log_contains"]
    assert recipe["steps"]["resume"]["command"][-1].endswith("interrupted-result.json")
    assert recipe["steps"]["first_valid_run"]["command"][2:4] == [
        "scripts.release.qualification_coverage", "--engine",
    ]
    assert "llm" in recipe["coverage"]["workloads"]
    assert "emb" in recipe["coverage"]["workloads"]
    assert "sustained" in recipe["coverage"]["workloads"]


def test_recipe_requires_both_lifecycle_versions(tmp_path):
    with pytest.raises(ValueError, match="are required"):
        build_recipe(
            target_id="macos-m5-pro-llamacpp-metal", root=tmp_path,
            output=tmp_path / "evidence", baseline_version="", target_version="b1",
            accelerator_identity="MacBook Pro / M5 Pro",
        )


def test_vllm_recipe_uses_private_cache_and_acknowledges_experimental_engine(tmp_path):
    recipe = build_recipe(
        target_id="ryzen-ai-halo-vllm-rocm", root=tmp_path,
        output=tmp_path / "evidence", baseline_version="0.26.0+rocm700",
        target_version="0.27.1+rocm723",
        accelerator_identity="AMD Radeon 8060S",
    )
    assert recipe["environment"]["HF_HOME"].endswith("qualification-vllm-cache")
    assert "scripts.release.qualification_coverage" in \
        recipe["steps"]["first_valid_run"]["command"]
    assert "vllmbench" in recipe["coverage"]["workloads"]
    assert "img" not in recipe["coverage"]["workloads"]


def test_llamacpp_recipe_covers_image_and_native_benchmark_workloads(tmp_path):
    recipe = build_recipe(
        target_id="macos-m5-pro-llamacpp-metal", root=tmp_path,
        output=tmp_path / "evidence", baseline_version="b1", target_version="b2",
        accelerator_identity="M5 Pro",
    )
    assert {"img", "llamabench", "llamabenchconc"} <= set(recipe["coverage"]["workloads"])
    assert recipe["coverage"]["models"] == [
        "gemma3:1b-it-q4_K_M", "nomic-embed-text", "sd15",
    ]


def test_recipe_preserves_virtualenv_python_symlink_path(tmp_path):
    venv_python = tmp_path / "qualification-env" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to("/usr/bin/python3")
    recipe = build_recipe(
        target_id="macos-m5-pro-llamacpp-metal", root=tmp_path,
        output=tmp_path / "evidence", baseline_version="b1", target_version="b2",
        python_executable=str(venv_python), accelerator_identity="M5 Pro",
    )
    assert recipe["steps"]["install"]["command"][0] == str(venv_python)


def test_recipe_rejects_a_different_accelerator_on_same_backend(tmp_path):
    with pytest.raises(ValueError, match="requires accelerator identity"):
        build_recipe(
            target_id="intel-arc-windows-llamacpp-vulkan", root=tmp_path,
            output=tmp_path / "evidence", baseline_version="b1", target_version="b2",
            accelerator_identity="AMD Radeon RX 9070 XT",
        )
