import pytest

from scripts.release.qualification_auto import (
    PINNED_VERSIONS, automatic_recipes, detected_targets, target_versions,
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
    ("Linux", "x86_64", "Ryzen AI Max+ 395\nAMD Radeon 8060S", False,
     ["ryzen-ai-halo-llamacpp-rocm", "ryzen-ai-halo-vllm-rocm"]),
    ("Linux", "aarch64", "NVIDIA GB10", False,
     ["dgx-spark-llamacpp-cuda", "dgx-spark-vllm-cuda"]),
])
def test_target_detection_selects_every_applicable_runtime(
        system, machine, identity, wsl, expected):
    assert detected_targets(system, machine, identity, wsl=wsl) == expected


def test_pinned_versions_are_complete_for_every_automatic_target():
    targets = [
        "macos-m5-pro-llamacpp-metal", "geforce-wsl2-vllm-cuda",
        "ryzen-ai-halo-vllm-rocm", "dgx-spark-vllm-cuda",
    ]
    assert all(all(target_versions(target)) for target in targets)
    assert PINNED_VERSIONS["vllm-rocm"][1] == "0.27.1+rocm723"


def test_automatic_recipe_records_detected_identity_and_pinned_target(tmp_path):
    recipes = automatic_recipes(
        tmp_path, tmp_path / "evidence", system="Darwin", machine="arm64",
        hostname="MacBook Pro\nM5 Pro 48 GB", wsl=False,
    )
    output, recipe = recipes[0]
    assert recipe["target"]["runtime_version"] == PINNED_VERSIONS["llamacpp"][1]
    assert recipe["target"]["accelerator"] == "MacBook Pro\nM5 Pro 48 GB"
    assert PINNED_VERSIONS["llamacpp"][1] in output.name
