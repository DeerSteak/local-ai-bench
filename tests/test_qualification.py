import pytest

from scripts.release.qualification import (
    QUALIFICATION_MATRIX, derive_image_support_level, derive_support_level, engine_selection_label,
    engine_support_profile, experimental_acknowledgement_required,
    experimental_engine_ack_error, platform_name,
    qualification_entry, qualification_is_stale, qualification_rows,
    validate_qualification_entry,
    validate_qualification_matrix,
)


def entry(**overrides):
    value = {
        "id": "linux-x86_64-llamacpp-cuda", "platform": "linux",
        "architecture": "x86_64", "runtime": "llamacpp", "runtime_version": "b6000",
        "backend": "cuda", "qualified_at": "2026-08-18", "suite_version": "6.0-pre8",
        "accelerator": "NVIDIA GeForce RTX 5090",
        "coverage": {
            "workloads": [
                "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
                "conc_tool", "conc_chat", "sustained", "llamabench", "llamabenchconc", "img",
            ],
            "models": ["gemma3:1b-it-q4_K_M", "nomic-embed-text", "sd15"],
            "notes": "Smallest-model functional coverage.",
        },
        "evidence": ["results_qualification_linux-x86_64-llamacpp-cuda.json"],
    }
    return {**value, **overrides}


def test_support_level_is_derived_from_complete_stale_and_absent_evidence():
    assert derive_support_level(entry(), "6.0-pre8") == "supported"
    assert derive_support_level(entry(suite_version="6.0"), "6.2") == "experimental"
    assert derive_support_level(None, "6.0-pre8") == "unverified"


def test_vllm_support_requires_its_smallest_tool_capable_model():
    evidence = entry(runtime="vllm", coverage={
        "workloads": [
            "llm", "conv", "emb", "mcq", "math", "reasoning", "code", "tool",
            "conc_tool", "conc_chat", "sustained", "vllmbench",
        ],
        "models": ["granite4.1:3b-q4_K_M", "nomic-embed-text"],
        "notes": "Smallest complete vLLM model coverage.",
    })
    assert derive_support_level(evidence, "6.0-pre8") == "supported"
    evidence["coverage"]["models"][0] = "gemma3:1b-it-q4_K_M"
    assert derive_support_level(evidence, "6.0-pre8") == "unverified"


def test_reviewed_macos_m5_pro_qualification_is_supported():
    evidence = qualification_entry(
        "macos", "arm64", "llamacpp", "metal", "b10488",
        accelerator="MacBook Pro\nM5 Pro 48 GB",
    )
    assert evidence == QUALIFICATION_MATRIX[0]
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_geforce_wsl2_qualification_is_supported():
    evidence = qualification_entry(
        "wsl2", "x86_64", "llamacpp", "cuda", "b10488",
        accelerator=(
            "Intel(R) Core(TM) Ultra 7 270K Plus\n"
            "NVIDIA GeForce RTX 5060 Ti 55 GB"
        ),
    )
    assert evidence == QUALIFICATION_MATRIX[1]
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_radeon_wsl2_qualification_is_supported_with_images():
    evidence = qualification_entry(
        "wsl2", "x86_64", "llamacpp", "rocm", "0.1.2-dev",
        accelerator=(
            "AMD Ryzen 7 5800XT 8-Core Processor\n"
            "AMD Radeon RX 9060 XT 31 GB"
        ),
    )
    assert evidence is not None and evidence["id"] == "radeon-wsl2-llamacpp-rocm"
    assert derive_support_level(evidence, "6.0-pre8") == "supported"
    assert derive_image_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_geforce_windows_qualification_is_supported():
    evidence = qualification_entry(
        "windows", "x86_64", "llamacpp", "cuda", "0.1.2-dev",
        accelerator="NVIDIA GeForce RTX 5060 Ti / 31.8574 GB VRAM",
    )
    assert evidence is not None and evidence["id"] == "geforce-windows-llamacpp-cuda"
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_radeon_8060s_windows_qualification_is_supported():
    evidence = qualification_entry(
        "windows", "x86_64", "llamacpp", "vulkan", "0.1.2-dev",
        accelerator=(
            "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S / 127 GB RAM\n"
            "AMD Radeon(TM) 8060S Graphics"
        ),
    )
    assert evidence is not None and evidence["id"] == \
        "radeon-8060s-windows-llamacpp-vulkan"
    assert derive_support_level(evidence, "6.0-pre8") == "supported"
    assert derive_image_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_dgx_spark_llamacpp_qualification_is_supported():
    evidence = qualification_entry(
        "linux", "aarch64", "llamacpp", "cuda", "0.1.2-dev",
        accelerator="NVIDIA GB10 122 GB",
    )
    assert evidence is not None and evidence["id"] == "dgx-spark-llamacpp-cuda"
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_dgx_spark_vllm_qualification_is_supported():
    evidence = qualification_entry(
        "linux", "aarch64", "vllm", "cuda", "0.27.1",
        accelerator="NVIDIA GB10 122 GB",
    )
    assert evidence is not None and evidence["id"] == "dgx-spark-vllm-cuda"
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_reviewed_ryzen_ai_halo_llamacpp_runtime_is_supported_without_images():
    evidence = qualification_entry(
        "linux", "x86_64", "llamacpp", "rocm", "0.1.2-dev",
        accelerator="AMD Ryzen AI MAX+ 395 w/ Radeon 8060S 125 GB",
    )
    assert evidence is not None and evidence["id"] == "ryzen-ai-halo-llamacpp-rocm"
    assert derive_support_level(evidence, "6.0-pre8") == "supported"
    assert derive_image_support_level(evidence, "6.0-pre8") == "unverified"


@pytest.mark.parametrize(("runtime", "runtime_version", "entry_id"), [
    ("llamacpp", "0.1.2-dev", "geforce-rtx-5090-wsl2-llamacpp-cuda"),
    ("vllm", "0.27.1", "geforce-rtx-5090-wsl2-vllm-cuda"),
])
def test_reviewed_geforce_rtx_5090_wsl2_qualifications_are_supported(
        runtime, runtime_version, entry_id):
    evidence = qualification_entry(
        "wsl2", "x86_64", runtime, "cuda", runtime_version,
        accelerator=(
            "AMD Ryzen 7 9850X3D 8-Core Processor\n"
            "NVIDIA GeForce RTX 5090 51 GB"
        ),
    )
    assert evidence is not None and evidence["id"] == entry_id
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_staleness_downgrades_at_release_boundary():
    evidence = entry(suite_version="6.0")
    assert qualification_is_stale(evidence, "6.1") is False
    assert qualification_is_stale(evidence, "6.2") is True
    assert derive_support_level(evidence, "6.2") == "experimental"


def test_support_level_cannot_be_set_in_evidence():
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_qualification_entry(entry(support_level="supported"))


def test_qualification_evidence_must_be_an_ordinary_result_json():
    with pytest.raises(ValueError, match="ordinary result JSON"):
        validate_qualification_entry(entry(evidence=["qualification-result.lab.zip"]))


def test_runtime_support_does_not_require_image_coverage():
    evidence = entry()
    evidence["coverage"]["workloads"].remove("img")
    evidence["coverage"]["models"].remove("sd15")
    assert derive_support_level(evidence, "6.0-pre8") == "supported"
    assert derive_image_support_level(evidence, "6.0-pre8") == "unverified"


def test_image_support_requires_image_workload_and_model_evidence():
    evidence = entry()
    assert derive_image_support_level(evidence, "6.0-pre8") == "supported"
    evidence["coverage"]["models"].remove("sd15")
    assert derive_image_support_level(evidence, "6.0-pre8") == "unverified"
    assert derive_image_support_level(None, "6.0-pre8") == "unverified"


def test_matrix_rejects_duplicate_runtime_identity():
    first = entry()
    second = entry(id="other")
    with pytest.raises(ValueError, match="runtime identities"):
        validate_qualification_matrix([first, second])


def test_wsl2_is_a_distinct_platform_and_requires_linux():
    assert platform_name("Linux") == "linux"
    assert platform_name("Linux", wsl=True) == "wsl2"
    with pytest.raises(ValueError, match="requires a Linux"):
        platform_name("Windows", wsl=True)


def test_matrix_rows_default_missing_targets_to_unverified():
    targets = [
        {"platform": "linux", "architecture": "x86_64", "runtime": "llamacpp",
         "backend": "cuda", "accelerator": "NVIDIA GeForce"},
        {"platform": "wsl2", "architecture": "x86_64", "runtime": "vllm",
         "backend": "cuda", "accelerator": "NVIDIA GeForce RTX 5090"},
    ]
    rows = qualification_rows("6.0-pre8", targets=targets, entries=[entry()])
    assert [row["support_level"] for row in rows] == ["supported", "unverified"]
    assert [row["image_support_level"] for row in rows] == ["supported", "unverified"]
    assert rows[1]["platform"] == "wsl2" and rows[1]["qualified_at"] is None


def test_runtime_lookup_can_require_the_exact_qualified_version():
    evidence = entry()
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b6000", [evidence],
    ) == evidence
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b7000", [evidence],
    ) is None


def test_runtime_lookup_normalizes_architecture_aliases_on_both_sides():
    evidence = entry(architecture="aarch64")
    assert qualification_entry(
        "linux", "arm64", "llamacpp", "cuda", "b6000", [evidence],
        accelerator="NVIDIA GeForce RTX 5090",
    ) == evidence


def test_runtime_lookup_distinguishes_accelerators_with_the_same_backend():
    evidence = entry(accelerator="AMD Radeon")
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b6000", [evidence],
        accelerator="Intel Arc",
    ) is None


def test_runtime_lookup_does_not_treat_gpu_model_as_a_complete_host_identity():
    evidence = entry(
        backend="rocm",
        accelerator="AMD Ryzen AI MAX+395 w/ Radeon 8060S\nAMD Ryzen AI Max+ 395 125 GB",
    )
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "rocm", "b6000", [evidence],
        accelerator="Radeon 8060S",
    ) is None


def test_runtime_lookup_rejects_accelerator_substrings_across_products():
    evidence = entry(accelerator="NVIDIA GeForce RTX 5090")
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b6000", [evidence],
        accelerator="NVIDIA GeForce RTX 5090 Laptop GPU",
    ) is None


def test_runtime_lookup_normalizes_accelerator_case_and_whitespace():
    evidence = entry(accelerator="NVIDIA GeForce RTX 5090\n32 GB")
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b6000", [evidence],
        accelerator="  nvidia geforce rtx 5090   32 gb ",
    ) == evidence


def test_execution_support_profile_requires_exact_runtime_evidence():
    evidence = entry()
    supported = engine_support_profile(
        system="Linux", architecture="AMD64", wsl=False, runtime="llamacpp",
        runtime_version="b6000", backend="cuda", current_version="6.0-pre8",
        entries=[evidence],
    )
    unverified = engine_support_profile(
        system="Linux", architecture="x86_64", wsl=False, runtime="llamacpp",
        runtime_version="b6001", backend="cuda", current_version="6.0-pre8",
        entries=[evidence],
    )
    assert supported["support_level"] == "supported"
    assert supported["qualification_id"] == evidence["id"]
    assert unverified["support_level"] == "unverified"


def test_engine_selection_labels_and_vllm_acknowledgement_follow_derived_support():
    unverified = {"support_level": "unverified"}
    supported = {"support_level": "supported"}
    assert engine_selection_label("llamacpp", unverified) == "llama.cpp — Unverified"
    assert engine_selection_label("vllm", unverified) == (
        "vllm — Experimental · Unverified qualification"
    )
    assert experimental_acknowledgement_required(
        ["llamacpp", "vllm"], {"llamacpp": unverified, "vllm": unverified},
    ) is True
    assert experimental_acknowledgement_required(
        ["vllm"], {"vllm": supported},
    ) is False
    error = experimental_engine_ack_error(["vllm"], False)
    assert error is not None and "--ack-experimental-engine" in error
    assert experimental_engine_ack_error(["vllm"], True) is None
    assert experimental_engine_ack_error(["llamacpp"], False) is None
