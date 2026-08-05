"""Reference-result evidence tiers and lifecycle decisions."""

from datetime import datetime, timezone


EVIDENCE_TIERS = ("verified", "vendor", "community", "rejected")


def classify_evidence(record):
    """Classify provenance without promoting a submission from reputation alone."""
    if record.get("removed_at") or record.get("rejected_reason"):
        return "rejected"
    if record.get("verification", {}).get("status") == "verified":
        return "verified"
    if record.get("source_type") == "vendor":
        return "vendor"
    return "community"


def recommendation_evidence_eligible(record, now=None):
    """Accept only current verified evidence with exact required identities."""
    if classify_evidence(record) != "verified":
        return False
    required = {"methodology_id", "hardware_id", "model_id", "runtime_id", "environment_id"}
    if not required <= set(record.get("identity", {})):
        return False
    expires_at = record.get("expires_at")
    if not expires_at:
        return False
    current = now or datetime.now(timezone.utc)
    return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) > current


def supersede_evidence(record, replacement_id, changed_at):
    """Return a correction record while preserving the original identity."""
    if not replacement_id or replacement_id == record.get("id"):
        raise ValueError("correction requires a different replacement identity")
    updated = dict(record)
    updated["superseded_by"] = replacement_id
    updated["superseded_at"] = changed_at
    return updated
