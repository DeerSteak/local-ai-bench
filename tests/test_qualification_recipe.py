import pytest

from scripts.release.qualification_automation import execution_recipe_gaps
from scripts.release.qualification_recipe import TARGETS, build_recipe


@pytest.mark.parametrize("target_id", TARGETS)
def test_every_declared_target_generates_a_complete_recipe(tmp_path, target_id):
    recipe = build_recipe(
        target_id=target_id, root=tmp_path, output=tmp_path / "evidence",
        baseline_version="baseline", target_version="target",
        python_executable="/usr/bin/python3",
    )
    assert execution_recipe_gaps(recipe) == []
    assert recipe["target"]["id"] == target_id
    assert recipe["steps"]["cancellation"]["interrupt_when_log_contains"]
    assert recipe["steps"]["resume"]["command"][-1].endswith("interrupted-result.json")


def test_recipe_requires_a_real_version_transition(tmp_path):
    with pytest.raises(ValueError, match="must be distinct"):
        build_recipe(
            target_id="macos-m5-pro-llamacpp-metal", root=tmp_path,
            output=tmp_path / "evidence", baseline_version="b1", target_version="b1",
        )


def test_vllm_recipe_uses_private_cache_and_acknowledges_experimental_engine(tmp_path):
    recipe = build_recipe(
        target_id="ryzen-ai-halo-vllm-rocm", root=tmp_path,
        output=tmp_path / "evidence", baseline_version="0.26.0+rocm700",
        target_version="0.27.1+rocm723",
    )
    assert recipe["environment"]["HF_HOME"].endswith("qualification-vllm-cache")
    assert "--ack-experimental-engine" in recipe["steps"]["first_valid_run"]["command"]
