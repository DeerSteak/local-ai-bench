import json
from pathlib import Path

import pytest

from scripts.results.acceptance_policy import evaluate_policy, load_policy, validate_policy
from scripts.results.acceptance_policy_cli import main


def policy(**rule_changes):
    rule = {
        "id": "llm-2k-tps", "section": "llm", "model": "golden", "case": "2K",
        "metric": "tps_mean", "operator": "at_least", "threshold": 49.0,
        "minimum_evidence": 2,
    }
    rule.update(rule_changes)
    return {"schema_version": 1, "name": "Vendor launch gate",
            "methodology_profile": "neutral-v1", "rules": [rule]}


def policy_v2(**rule_changes):
    candidate = policy(**rule_changes)
    candidate["schema_version"] = 2
    candidate["rules"][0].update({
        "tolerance_pct": 3.0,
        "evidence_requirement": "single_run",
    })
    return candidate


def result():
    return {
        "run": {"plan": {"effective_config": {"methodology_profile": "neutral-v1"}}},
        "llm": {"golden": {"2K": {"tps_mean": 50.0, "valid_runs": 2}}},
    }


def test_acceptance_passes_only_named_threshold_with_enough_evidence():
    evaluation = evaluate_policy(result(), policy())
    assert evaluation["decision"] == "accepted"
    assert evaluation["rules"] == [{
        "id": "llm-2k-tps", "status": "pass", "actual": 50.0,
        "threshold": 49.0, "evidence": 2,
    }]


@pytest.mark.parametrize(("change", "status"), [
    ({"threshold": 51.0}, "fail"),
    ({"minimum_evidence": 3}, "insufficient_evidence"),
    ({"model": "missing"}, "missing"),
])
def test_acceptance_rejects_failed_missing_and_insufficient_evidence(change, status):
    evaluation = evaluate_policy(result(), policy(**change))
    assert evaluation["decision"] == "rejected"
    assert evaluation["rules"][0]["status"] == status


def test_acceptance_rejects_incompatible_or_unrecorded_methodology():
    for actual in ("tuned-v1", None):
        value = result()
        value["run"]["plan"]["effective_config"]["methodology_profile"] = actual
        evaluation = evaluate_policy(value, policy())
        assert evaluation["rules"][0]["status"] == "incompatible_methodology"


def test_accuracy_and_image_rules_use_their_real_evidence_shapes():
    value = result()
    value.update({
        "mcq": {"golden": {"accuracy_pct": 75.0, "total": 100}},
        "images": {"flux": {"resolutions": {"1024x1024": {
            "sec_per_image_mean": 8.0, "n_runs": 3,
        }}}},
    })
    rules = policy()["rules"] = [
        {"id": "quality", "section": "mcq", "model": "golden", "case": None,
         "metric": "accuracy_pct", "operator": "at_least", "threshold": 70.0,
         "minimum_evidence": 100},
        {"id": "image", "section": "images", "model": "flux", "case": "1024x1024",
         "metric": "sec_per_image_mean", "operator": "at_most", "threshold": 10.0,
         "minimum_evidence": 3},
    ]
    candidate = policy()
    candidate["rules"] = rules
    assert evaluate_policy(value, candidate)["decision"] == "accepted"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version=3),
    lambda p: p.update(rules=[]),
    lambda p: p["rules"][0].update(metric="unknown"),
    lambda p: p["rules"][0].update(threshold=float("nan")),
    lambda p: p["rules"][0].update(case=None),
])
def test_policy_validation_rejects_malformed_or_unsafe_rules(mutation):
    candidate = policy()
    mutation(candidate)
    with pytest.raises(ValueError):
        validate_policy(candidate)


def test_policy_loader_and_cli_use_deterministic_machine_readable_result(tmp_path, capsys):
    result_path = tmp_path / "result.json"
    policy_path = tmp_path / "policy.json"
    result_path.write_text(json.dumps(result()), encoding="utf-8")
    policy_path.write_text(json.dumps(policy()), encoding="utf-8")
    assert load_policy(policy_path)["name"] == "Vendor launch gate"
    assert main([str(result_path), str(policy_path)]) == 0
    assert '"decision": "accepted"' in capsys.readouterr().out


def test_schema_two_accepts_a_value_inside_practical_tolerance():
    candidate = result()
    candidate["llm"]["golden"]["2K"]["tps_mean"] = 48.0
    evaluation = evaluate_policy(candidate, policy_v2())
    assert evaluation["decision"] == "accepted"
    assert evaluation["rules"][0]["status"] == "pass_within_tolerance"


def test_reproducible_evidence_requirement_is_inconclusive_for_one_run():
    candidate = policy_v2()
    candidate["rules"][0]["evidence_requirement"] = "repeated_trials"
    evaluation = evaluate_policy(result(), candidate)
    assert evaluation["decision"] == "inconclusive"
    assert evaluation["rules"][0]["status"] == "inconclusive"


@pytest.mark.parametrize("change", [
    {"tolerance_pct": -1},
    {"evidence_requirement": "pretend_significant"},
])
def test_schema_two_rejects_invalid_noise_policy_fields(change):
    candidate = policy_v2()
    candidate["rules"][0].update(change)
    with pytest.raises(ValueError):
        validate_policy(candidate)
