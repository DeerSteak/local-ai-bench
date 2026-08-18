import pytest

from scripts.release.qualification import (
    QUALIFICATION_LIFECYCLE, derive_support_level, platform_name,
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
        "lifecycle": lifecycle,
        "known_failures": [
            {"step": step, "detail": f"{state} during qualification"}
            for step, state in lifecycle.items() if state != "passed"
        ],
        "evidence": ["qualification/linux-x86_64-llamacpp-cuda.json"],
    }
    return {**value, **overrides}


def test_support_level_is_derived_from_complete_partial_and_absent_evidence():
    assert derive_support_level(entry(), "6.0-pre8") == "supported"
    assert derive_support_level(entry({"rollback": "failed"}), "6.0-pre8") == "experimental"
    assert derive_support_level(entry({step: "not_tested" for step in QUALIFICATION_LIFECYCLE}),
                                "6.0-pre8") == "unverified"
    assert derive_support_level(None, "6.0-pre8") == "unverified"


def test_staleness_downgrades_at_release_boundary():
    evidence = entry(suite_version="6.0")
    assert qualification_is_stale(evidence, "6.1") is False
    assert qualification_is_stale(evidence, "6.2") is True
    assert derive_support_level(evidence, "6.2") == "experimental"


def test_support_level_cannot_be_set_in_evidence():
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_qualification_entry(entry(support_level="supported"))


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
         "backend": "cuda"},
        {"platform": "wsl2", "architecture": "x86_64", "runtime": "vllm",
         "backend": "cuda"},
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
