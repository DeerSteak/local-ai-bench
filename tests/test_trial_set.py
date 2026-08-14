import json

import pytest

from scripts.results.trial_set import (
    aggregate_trials, analyze_trial_metric, build_trial_set, monotonic_drift,
    trial_set_compatibility,
)
from scripts.results.trial_set_cli import main
from tests.test_result_history import result


def trials(values, *, hostname="system"):
    return [result(tps=value, hostname=hostname, started=f"2026-01-{index + 1:02d}T00:00:00Z")
            for index, value in enumerate(values)]


def test_aggregate_requires_five_trials_before_emitting_an_interval():
    assert aggregate_trials([1.0, 1.1])["interval"] is None
    aggregate = aggregate_trials([1.0, 1.1, 0.9, 1.05, 0.95])
    assert aggregate["trial_count"] == 5
    assert aggregate["interval_method"] == "student_t_95"
    assert len(aggregate["interval"]) == 2


def test_drift_detector_flags_monotonic_decline_but_not_flat_noise():
    assert monotonic_drift([5.0, 4.0, 3.0]) == "declining"
    assert monotonic_drift([5.0, 5.1, 4.9, 5.0]) == "none"
    assert monotonic_drift([5.0, 5.0, 5.0]) == "none"


def test_paired_trial_verdicts_cover_improved_regressed_unchanged_and_inconclusive():
    baseline = [100.0, 101.0, 99.0, 100.5, 99.5]
    assert analyze_trial_metric("tps_mean", baseline,
                                [110.0, 111.1, 108.9, 110.55, 109.45], paired=True)["verdict"] == "improved"
    assert analyze_trial_metric("tps_mean", baseline,
                                [90.0, 90.9, 89.1, 90.45, 89.55], paired=True)["verdict"] == "regressed"
    assert analyze_trial_metric("tps_mean", baseline,
                                [101.0, 102.01, 99.99, 101.505, 100.495], paired=True)["verdict"] == "unchanged"
    assert analyze_trial_metric("tps_mean", baseline[:2], baseline[:2], paired=True)["verdict"] == "inconclusive"


def test_drift_forces_an_inconclusive_verdict():
    analysis = analyze_trial_metric(
        "tps_mean", [100, 99, 98, 97, 96], [110, 109, 108, 107, 106], paired=True)
    assert analysis["verdict"] == "inconclusive"
    assert analysis["baseline"]["drift"] == "declining"


def test_unequal_trial_counts_use_an_independent_welch_interval():
    analysis = analyze_trial_metric(
        "tps_mean",
        [99.8, 100.2, 100.0, 99.9, 100.1],
        [109.8, 110.2, 110.0, 109.9, 110.1, 110.05],
        paired=False,
    )
    assert analysis["comparison_mode"] == "independent"
    assert analysis["interval_method"] == "welch_t_95"
    assert analysis["verdict"] == "improved"


def test_trial_pool_rejects_methodology_and_hardware_mismatches():
    mismatched_method = trials([50, 51])
    mismatched_method[1]["run"]["plan"]["effective_config"]["runs"] = 5
    assert not trial_set_compatibility(mismatched_method)["compatible"]
    assert trial_set_compatibility(trials([50], hostname="a") + trials([51], hostname="b")) == {
        "compatible": False, "incompatible_fields": ["hardware_identity"],
    }


def test_build_trial_set_pairs_identical_case_sequences_and_rejects_incompatible_inputs():
    artifact = build_trial_set(
        trials([50, 51, 49, 50.5, 49.5]), trials([55, 56.1, 53.9, 55.55, 54.45]))
    assert artifact["comparison_mode"] == "paired"
    assert artifact["rows"]
    bad = trials([55, 56, 54, 55.5, 54.5], hostname="other")
    with pytest.raises(ValueError, match="hardware_identity"):
        build_trial_set(trials([50, 51, 49, 50.5, 49.5]), bad)


def test_trial_set_cli_writes_a_versioned_artifact(tmp_path):
    baseline_paths = []
    candidate_paths = []
    for label, values, paths in (
        ("baseline", [50, 51, 49, 50.5, 49.5], baseline_paths),
        ("candidate", [55, 56.1, 53.9, 55.55, 54.45], candidate_paths),
    ):
        for index, trial in enumerate(trials(values)):
            path = tmp_path / f"{label}-{index}.json"
            path.write_text(json.dumps(trial), encoding="utf-8")
            paths.append(path)
    output = tmp_path / "trials.json"
    argv = ["--baseline", *(str(path) for path in baseline_paths),
            "--candidate", *(str(path) for path in candidate_paths), "--out", str(output)]
    assert main(argv) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["comparison_mode"] == "paired"
    assert len(artifact["source_sha256"]["baseline"]) == 5
