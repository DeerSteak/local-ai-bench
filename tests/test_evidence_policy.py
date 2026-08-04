from datetime import datetime, timezone

import pytest

from scripts.results.evidence_policy import classify_evidence, recommendation_evidence_eligible, supersede_evidence


IDENTITY = {
    "methodology_id": "method-1", "hardware_id": "hardware-1", "model_id": "model-1",
    "runtime_id": "runtime-1", "environment_id": "environment-1",
}


def record(**changes):
    value = {
        "id": "result-1", "source_type": "community",
        "verification": {"status": "verified"}, "identity": IDENTITY,
        "expires_at": "2027-01-01T00:00:00Z",
    }
    value.update(changes)
    return value


def test_only_explicit_verification_reaches_verified_tier():
    assert classify_evidence(record()) == "verified"
    assert classify_evidence(record(verification={"status": "pending"})) == "community"
    assert classify_evidence(record(source_type="vendor", verification={"status": "pending"})) == "vendor"


def test_rejection_or_removal_overrides_source_and_verification():
    assert classify_evidence(record(rejected_reason="tampered")) == "rejected"
    assert classify_evidence(record(removed_at="2026-01-01T00:00:00Z")) == "rejected"


def test_supported_recommendations_require_current_complete_verified_identity():
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    assert recommendation_evidence_eligible(record(), now) is True
    assert recommendation_evidence_eligible(record(expires_at="2026-01-01T00:00:00Z"), now) is False
    assert recommendation_evidence_eligible(record(expires_at=None), now) is False
    assert recommendation_evidence_eligible(record(identity={"hardware_id": "hardware-1"}), now) is False
    assert recommendation_evidence_eligible(record(verification={"status": "pending"}), now) is False


def test_correction_preserves_original_and_points_to_replacement():
    original = record()
    corrected = supersede_evidence(original, "result-2", "2026-08-04T00:00:00Z")
    assert original.get("superseded_by") is None
    assert corrected["superseded_by"] == "result-2"
    with pytest.raises(ValueError, match="different replacement"):
        supersede_evidence(original, "result-1", "2026-08-04T00:00:00Z")
