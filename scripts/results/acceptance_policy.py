"""Versioned evidence-threshold policies with explicit rejection reasons."""

import json
import math
from pathlib import Path

from scripts.results.result_store import as_dict, validate_json_data


POLICY_SCHEMA_VERSION = 2
SUPPORTED_POLICY_SCHEMAS = {1, 2}
OPERATORS = {"at_least", "at_most"}
EVIDENCE_REQUIREMENTS = {"single_run", "repeated_trials"}
SECTION_METRICS = {
    "llm": {"tps_mean", "ttft_mean_sec"},
    "llm_conversation": {"tps_mean", "client_ttft_mean_sec", "server_prompt_mean_sec"},
    "llm_cached": {"tps_mean", "prefill_tps_mean", "server_prompt_mean_sec"},
    "embeddings": {"chunks_per_sec_mean"},
    "images": {"sec_per_image_mean"},
    "mcq": {"accuracy_pct"}, "math": {"accuracy_pct"},
    "reasoning": {"accuracy_pct"}, "code": {"accuracy_pct"},
    "tool": {"accuracy_pct"},
    "concurrency_tool": {"aggregate_tps"}, "concurrency_chat": {"aggregate_tps"},
}
CASE_REQUIRED = {"llm", "llm_cached", "llm_conversation", "images", "concurrency_tool", "concurrency_chat"}


def validate_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or policy.get("schema_version") not in SUPPORTED_POLICY_SCHEMAS:
        raise ValueError("unsupported or missing acceptance-policy schema")
    if set(policy) != {"schema_version", "name", "methodology_profile", "rules"}:
        raise ValueError("acceptance policy contains unknown fields")
    if not isinstance(policy.get("name"), str) or not policy["name"].strip():
        raise ValueError("acceptance policy requires a name")
    profile = policy.get("methodology_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("acceptance policy requires a methodology profile")
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("acceptance policy requires at least one rule")
    rule_ids = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("acceptance policy rules must be objects")
        required = {"id", "section", "model", "case", "metric", "operator", "threshold",
                    "minimum_evidence"}
        if policy["schema_version"] == 2:
            required |= {"tolerance_pct", "evidence_requirement"}
        if set(rule) != required:
            raise ValueError(f"acceptance rule fields must be exactly: {sorted(required)}")
        if not isinstance(rule["id"], str) or not rule["id"] or rule["id"] in rule_ids:
            raise ValueError("acceptance rule IDs must be unique non-empty strings")
        rule_ids.add(rule["id"])
        section = rule["section"]
        if section not in SECTION_METRICS or rule["metric"] not in SECTION_METRICS[section]:
            raise ValueError(f"unsupported acceptance metric: {section}.{rule['metric']}")
        if not isinstance(rule["model"], str) or not rule["model"]:
            raise ValueError("acceptance rule requires a model")
        if (section in CASE_REQUIRED) != isinstance(rule["case"], str):
            raise ValueError(f"acceptance rule case shape does not match section: {section}")
        if isinstance(rule["case"], str) and not rule["case"]:
            raise ValueError("acceptance rule case must not be empty")
        if rule["operator"] not in OPERATORS:
            raise ValueError(f"unsupported acceptance operator: {rule['operator']}")
        threshold = rule["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                or not math.isfinite(threshold):
            raise ValueError("acceptance rule threshold must be finite")
        minimum = rule["minimum_evidence"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("acceptance rule minimum_evidence must be a positive integer")
        if policy["schema_version"] == 2:
            tolerance = rule["tolerance_pct"]
            if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                    or not math.isfinite(tolerance) or tolerance < 0):
                raise ValueError("acceptance rule tolerance_pct must be finite and non-negative")
            if rule["evidence_requirement"] not in EVIDENCE_REQUIREMENTS:
                raise ValueError("unsupported acceptance evidence requirement")
    return policy


def load_policy(path: Path) -> dict:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_json_data(policy)
    return validate_policy(policy)


def _methodology_profile(result: dict) -> str | None:
    if isinstance(result.get("methodology_profile"), str):
        return result["methodology_profile"]
    run = as_dict(result.get("run"))
    plan = as_dict(run.get("plan"))
    settings = as_dict(plan.get("effective_config"))
    return settings.get("methodology_profile")


def _evidence(result: dict, rule: dict) -> tuple[dict | None, int]:
    if (result.get("schema_version") == 1 and result.get("compatible") is True
            and isinstance(result.get("rows"), list)):
        parts = [rule["section"], rule["model"]]
        if rule["case"] is not None:
            parts.append(rule["case"])
        parts.append(rule["metric"])
        key = "/".join(parts)
        row = next((item for item in result["rows"]
                    if isinstance(item, dict) and item.get("key") == key), None)
        candidate = row.get("candidate") if isinstance(row, dict) else None
        if not isinstance(candidate, dict):
            return None, 0
        count = candidate.get("trial_count")
        evidence = {
            rule["metric"]: candidate.get("mean"),
            "_trial_interval": candidate.get("interval"),
            "_trial_drift": candidate.get("drift"),
        }
        return evidence, int(count or 0)
    section = result.get(rule["section"])
    model = section.get(rule["model"]) if isinstance(section, dict) else None
    if not isinstance(model, dict):
        return None, 0
    if rule["case"] is not None:
        cases = model.get("resolutions") if rule["section"] == "images" else model
        evidence = cases.get(rule["case"]) if isinstance(cases, dict) else None
    else:
        evidence = model
    if not isinstance(evidence, dict):
        return None, 0
    count = evidence.get("valid_runs", evidence.get("n_runs"))
    if count is None and rule["metric"] == "accuracy_pct":
        count = evidence.get("total")
    return evidence, int(count or 0)


def evaluate_policy(result: dict, policy: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("result must be a JSON object")
    validate_json_data(result)
    validate_policy(policy)
    actual_profile = _methodology_profile(result)
    expected_profile = policy["methodology_profile"]
    schema_version = policy["schema_version"]
    outcomes = []
    for rule in policy["rules"]:
        tolerance_pct = float(rule.get("tolerance_pct", 0.0))
        requirement = rule.get("evidence_requirement", "single_run")
        outcome = {"id": rule["id"], "status": "missing", "actual": None,
                   "threshold": rule["threshold"], "evidence": 0}
        if schema_version == 2:
            outcome.update({"tolerance_pct": tolerance_pct,
                            "evidence_requirement": requirement})
        if actual_profile != expected_profile:
            outcome["status"] = "incompatible_methodology"
        else:
            evidence, count = _evidence(result, rule)
            outcome["evidence"] = count
            value = evidence.get(rule["metric"]) if evidence else None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                outcome["status"] = "missing"
            elif count < rule["minimum_evidence"]:
                outcome.update({"status": "insufficient_evidence", "actual": value})
            elif requirement == "repeated_trials":
                interval = evidence.get("_trial_interval") if evidence else None
                drift = evidence.get("_trial_drift") if evidence else None
                if not (isinstance(interval, list) and len(interval) == 2
                        and all(isinstance(bound, (int, float)) for bound in interval)) \
                        or drift not in {"none"}:
                    outcome.update({"status": "inconclusive", "actual": value})
                else:
                    low, high = interval
                    threshold = rule["threshold"]
                    tolerance = abs(threshold) * tolerance_pct / 100
                    if rule["operator"] == "at_least":
                        status = "pass" if low >= threshold else \
                            "pass_within_tolerance" if low >= threshold - tolerance else \
                            "fail" if high < threshold - tolerance else "inconclusive"
                    else:
                        status = "pass" if high <= threshold else \
                            "pass_within_tolerance" if high <= threshold + tolerance else \
                            "fail" if low > threshold + tolerance else "inconclusive"
                    outcome.update({"status": status, "actual": value})
            else:
                threshold = rule["threshold"]
                passed = value >= threshold if rule["operator"] == "at_least" \
                    else value <= threshold
                tolerance = abs(threshold) * tolerance_pct / 100
                tolerated = value >= threshold - tolerance if rule["operator"] == "at_least" \
                    else value <= threshold + tolerance
                status = "pass" if passed else "pass_within_tolerance" if tolerated else "fail"
                outcome.update({"status": status, "actual": value})
        outcomes.append(outcome)
    passing = {"pass", "pass_within_tolerance"}
    if schema_version == 1:
        decision = "accepted" if all(item["status"] == "pass" for item in outcomes) else "rejected"
    elif any(item["status"] == "fail" for item in outcomes):
        decision = "rejected"
    elif all(item["status"] in passing for item in outcomes):
        decision = "accepted"
    else:
        decision = "inconclusive"
    return {
        "schema_version": schema_version,
        "policy": policy["name"],
        "decision": decision,
        "methodology_profile": {"expected": expected_profile, "actual": actual_profile},
        "rules": outcomes,
    }
