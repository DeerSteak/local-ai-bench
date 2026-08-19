import pytest

from scripts.release.qualification import (
    QUALIFICATION_LIFECYCLE, QUALIFICATION_MATRIX, derive_support_level, engine_selection_label,
    engine_support_profile, experimental_acknowledgement_required,
    experimental_engine_ack_error, platform_name,
    qualification_entry, qualification_is_stale, qualification_rows,
    validate_qualification_entry,
    validate_qualification_matrix,
)


def entry(states=None, **overrides):
    lifecycle = {step: "passed" for step in QUALIFICATION_LIFECYCLE}
    lifecycle.update(states or {})
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
        "lifecycle": lifecycle,
        "known_failures": [
            {"step": step, "detail": f"{state} during qualification"}
            for step, state in lifecycle.items() if state != "passed"
        ],
        "evidence": ["qualification/linux-x86_64-llamacpp-cuda/qualification-manifest.json"],
    }
    return {**value, **overrides}


def test_support_level_is_derived_from_complete_partial_and_absent_evidence():
    assert derive_support_level(entry(), "6.0-pre8") == "supported"
    assert derive_support_level(entry({"rollback": "failed"}), "6.0-pre8") == "experimental"
    assert derive_support_level(entry({step: "not_tested" for step in QUALIFICATION_LIFECYCLE}),
                                "6.0-pre8") == "unverified"
    assert derive_support_level(None, "6.0-pre8") == "unverified"


def test_reviewed_macos_m5_pro_qualification_is_supported():
    evidence = qualification_entry(
        "macos", "arm64", "llamacpp", "metal", "b10488",
        accelerator="MacBook Pro\nM5 Pro 48 GB",
    )
    assert evidence == QUALIFICATION_MATRIX[0]
    assert derive_support_level(evidence, "6.0-pre8") == "supported"


def test_staleness_downgrades_at_release_boundary():
    evidence = entry(suite_version="6.0")
    assert qualification_is_stale(evidence, "6.1") is False
    assert qualification_is_stale(evidence, "6.2") is True
    assert derive_support_level(evidence, "6.2") == "experimental"


def test_support_level_cannot_be_set_in_evidence():
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_qualification_entry(entry(support_level="supported"))


def test_complete_lifecycle_requires_the_verified_final_manifest():
    with pytest.raises(ValueError, match="final evidence manifest"):
        validate_qualification_entry(entry(evidence=["qualification-state.json"]))


def test_complete_lifecycle_without_required_workload_coverage_is_unverified():
    evidence = entry()
    evidence["coverage"]["workloads"].remove("img")
    assert derive_support_level(evidence, "6.0-pre8") == "unverified"


def test_partial_evidence_requires_every_gap_to_be_documented():
    evidence = entry({"rollback": "failed"})
    evidence["known_failures"] = []
    with pytest.raises(ValueError, match="every incomplete"):
        validate_qualification_entry(evidence)


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
         "backend": "cuda", "accelerator": "NVIDIA GeForce RTX 5090"},
        {"platform": "wsl2", "architecture": "x86_64", "runtime": "vllm",
         "backend": "cuda", "accelerator": "NVIDIA GeForce RTX 5090"},
    ]
    rows = qualification_rows("6.0-pre8", targets=targets, entries=[entry()])
    assert [row["support_level"] for row in rows] == ["supported", "unverified"]
    assert rows[1]["platform"] == "wsl2" and rows[1]["qualified_at"] is None


def test_runtime_lookup_can_require_the_exact_qualified_version():
    evidence = entry()
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b6000", [evidence],
    ) == evidence
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b7000", [evidence],
    ) is None


def test_runtime_lookup_distinguishes_accelerators_with_the_same_backend():
    evidence = entry(accelerator="AMD Radeon")
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "cuda", "b6000", [evidence],
        accelerator="Intel Arc",
    ) is None


def test_runtime_lookup_matches_ryzen_ai_halo_identity_by_gpu_model():
    evidence = entry(
        backend="rocm",
        accelerator="AMD Ryzen AI MAX+395 w/ Radeon 8060S\nAMD Ryzen AI Max+ 395 125 GB",
    )
    assert qualification_entry(
        "linux", "x86_64", "llamacpp", "rocm", "b6000", [evidence],
        accelerator="Radeon 8060S",
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
