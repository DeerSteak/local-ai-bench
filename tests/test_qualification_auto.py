import pytest

from scripts.release.qualification import QUALIFICATION_LIFECYCLE
from scripts.release.qualification_auto import (
    AUTOMATION_REVISION, PINNED_VERSIONS, automatic_recipes, detected_targets,
    evidence_revision, execution_summary, select_runtime_targets, target_versions,
)


@pytest.mark.parametrize(("system", "machine", "identity", "wsl", "expected"), [
    ("Darwin", "arm64", "MacBook Pro\nM5 Pro 48 GB", False,
     ["macos-m5-pro-llamacpp-metal"]),
    ("Windows", "AMD64", "CPU\nNVIDIA GeForce RTX 5090", False,
     ["geforce-windows-llamacpp-cuda"]),
    ("Windows", "AMD64", "CPU\nAMD Radeon RX 9070", False,
     ["radeon-windows-llamacpp-vulkan"]),
    ("Windows", "AMD64", "CPU\nIntel Arc Pro B65", False,
     ["intel-arc-windows-llamacpp-vulkan"]),
    ("Linux", "x86_64", "CPU\nNVIDIA GeForce RTX 5090", True,
     ["geforce-wsl2-llamacpp-cuda", "geforce-wsl2-vllm-cuda"]),
    ("Linux", "x86_64", "CPU\nAMD Radeon RX 9060 XT", True,
     ["radeon-wsl2-llamacpp-rocm", "radeon-wsl2-vllm-rocm"]),
    ("Linux", "x86_64", "Ryzen AI Max+ 395\nAMD Radeon 8060S", False,
     ["ryzen-ai-halo-llamacpp-rocm", "ryzen-ai-halo-vllm-rocm"]),
    ("Linux", "aarch64", "NVIDIA GB10", False,
     ["dgx-spark-llamacpp-cuda", "dgx-spark-vllm-cuda"]),
])
def test_target_detection_selects_every_applicable_runtime(
        system, machine, identity, wsl, expected):
    assert detected_targets(system, machine, identity, wsl=wsl) == expected


def test_pinned_versions_are_complete_for_every_automatic_target():
    assert AUTOMATION_REVISION == "v8"
    targets = [
        "macos-m5-pro-llamacpp-metal", "geforce-wsl2-vllm-cuda",
        "radeon-wsl2-vllm-rocm", "ryzen-ai-halo-vllm-rocm", "dgx-spark-vllm-cuda",
    ]
    assert all(all(target_versions(target)) for target in targets)
    assert PINNED_VERSIONS["vllm-rocm"][1] == "0.27.1+rocm723"


def test_only_native_windows_uses_fresh_post_ctrl_break_evidence():
    assert evidence_revision("geforce-windows-llamacpp-cuda") == "v9"
    assert evidence_revision("radeon-windows-llamacpp-vulkan") == "v9"
    assert evidence_revision("radeon-wsl2-llamacpp-rocm") == "v8"


def test_automatic_recipe_records_detected_identity_and_pinned_target(tmp_path):
    recipes = automatic_recipes(
        tmp_path, tmp_path / "evidence", system="Darwin", machine="arm64",
        hostname="MacBook Pro\nM5 Pro 48 GB", wsl=False,
    )
    output, recipe = recipes[0]
    assert recipe["target"]["runtime_version"] == PINNED_VERSIONS["llamacpp"][1]
    assert recipe["target"]["accelerator"] == "MacBook Pro\nM5 Pro 48 GB"
    assert PINNED_VERSIONS["llamacpp"][1] in output.name
    assert output.name.endswith(AUTOMATION_REVISION)


def test_vllm_only_selects_only_the_vllm_target(tmp_path):
    recipes = automatic_recipes(
        tmp_path, tmp_path / "evidence", system="Linux", machine="aarch64",
        hostname="NVIDIA GB10", wsl=False, vllm_only=True,
    )
    assert [recipe[1]["target"]["id"] for recipe in recipes] == ["dgx-spark-vllm-cuda"]


def test_vllm_only_rejects_a_machine_without_a_vllm_target():
    with pytest.raises(ValueError, match="no automatic vLLM qualification target"):
        select_runtime_targets(["macos-m5-pro-llamacpp-metal"], vllm_only=True)


def test_failed_execution_summary_points_directly_to_step_log(tmp_path):
    steps = {
        name: {"status": "pending", "detail": None, "log": None}
        for name in QUALIFICATION_LIFECYCLE
    }
    steps["install"] = {
        "status": "failed", "detail": "missing dependency", "log": "01-install.log",
    }
    state = {"steps": steps}
    assert execution_summary({"id": "mac"}, state, tmp_path) == {
        "target": "mac", "status": "failed", "failed_step": "install",
        "detail": "missing dependency", "log": str(tmp_path / "01-install.log"),
        "evidence_dir": str(tmp_path),
    }
