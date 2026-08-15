from scripts.results.significance import compare_metric, metric_evidence, practical_threshold_pct


def test_metric_evidence_preserves_missing_dispersion_as_unknown():
    evidence = metric_evidence({"tps_mean": 50.0, "valid_runs": 3}, "tps_mean", "tps_stdev")
    assert evidence == {"value": 50.0, "dispersion": None, "sample_count": 3}


def test_comparison_reports_recorded_uncertainty_and_practical_threshold():
    before = {"value": 50.0, "dispersion": 1.0, "sample_count": 3}
    after = {"value": 52.0, "dispersion": 1.2, "sample_count": 3}
    comparison = compare_metric("tps_mean", before, after)
    assert comparison["within_run_uncertainty"] == "recorded"
    assert comparison["practical_threshold_pct"] == 3.0
    assert comparison["clears_practical_threshold"] is True
    assert comparison["verdict"] == "repeated_trials_required"


def test_comparison_requires_both_dispersion_values_and_two_samples():
    before = {"value": 1.0, "dispersion": 0.0, "sample_count": 1}
    after = {"value": 1.1, "dispersion": None, "sample_count": 3}
    assert compare_metric("ttft_mean_sec", before, after)["within_run_uncertainty"] == "insufficient"


def test_threshold_families_use_qualified_noise_floors():
    assert practical_threshold_pct("ttft_mean_sec") == 8.0
    assert practical_threshold_pct("aggregate_tps") == 3.0
    assert practical_threshold_pct("sec_per_image_mean") == 3.0
    assert practical_threshold_pct("accuracy_pct") == 1.0
