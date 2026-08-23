import pytest

from scripts.results.variant_comparison import build_variant_comparison


def result(*, include_q8=True, missing_memory=False):
    identities = [
        {"tag": "demo:q4", "short": "demo-q4", "base_model": "demo", "variant": "Q4_K_M"},
    ]
    if include_q8:
        identities.append(
            {"tag": "demo:q8", "short": "demo-q8", "base_model": "demo", "variant": "Q8_0"},
        )
    data = {
        "run": {"plan": {"models": {"llm": identities}}},
        "llm": {
            "demo-q4": {"2K": {
                "tps_mean": 20.0,
                "memory": {"summary": {"process_rss_gb": {"peak_gb": 4.0}}},
                "power": {"energy_joules": 10.0},
            }},
            "demo-q8": {"2K": {
                "tps_mean": 15.0,
                "memory": {"summary": {"accelerator_memory_used_gb": {"peak_gb": 8.0}}},
                "power": {"energy_joules": 12.5},
            }},
        },
        "mcq": {
            "demo-q4": {"accuracy_pct": 70.0},
            "demo-q8": {"accuracy_pct": 72.0},
        },
    }
    if missing_memory:
        data["llm"]["demo-q8"]["2K"].pop("memory")
    return data


def test_variant_tradeoffs_are_relative_to_stated_reference():
    artifact = build_variant_comparison(
        result(), base_model="demo", reference_variant="Q4_K_M",
        performance_section="llm", case="2K", accuracy_section="mcq",
        quality_verdicts={"Q8_0": "improved"},
    )

    q4, q8 = artifact["variants"]
    assert artifact["reference_variant"] == "Q4_K_M"
    assert q4["quality"]["delta"] == 0
    assert q8["quality"]["delta"] == 2
    assert q8["throughput"]["delta"] == -25
    assert q8["memory"]["delta"] == 100
    assert q8["energy"]["delta"] == 25
    assert q8["quality_ranked"] is True


@pytest.mark.parametrize("verdict", ["unchanged", "inconclusive"])
def test_unchanged_or_inconclusive_quality_is_not_ranked(verdict):
    artifact = build_variant_comparison(
        result(), base_model="demo", reference_variant="Q4_K_M",
        performance_section="llm", case="2K", accuracy_section="mcq",
        quality_verdicts={"Q8_0": verdict},
    )
    assert artifact["variants"][1]["quality_verdict"] == verdict
    assert artifact["variants"][1]["quality_ranked"] is False


def test_missing_measurement_is_unavailable_not_zero():
    artifact = build_variant_comparison(
        result(missing_memory=True), base_model="demo", reference_variant="Q4_K_M",
        performance_section="llm", case="2K", accuracy_section="mcq",
    )
    assert artifact["variants"][1]["memory"] == {
        "value": None, "delta": None, "status": "unavailable",
    }


def test_single_variant_comparison_is_valid_reference_only():
    artifact = build_variant_comparison(
        result(include_q8=False), base_model="demo", reference_variant="Q4_K_M",
        performance_section="llm", case="2K", accuracy_section="mcq",
    )
    assert len(artifact["variants"]) == 1
    assert artifact["variants"][0]["reference"] is True


def test_missing_variant_and_invalid_verdict_are_rejected():
    with pytest.raises(ValueError, match="reference variant is not present"):
        build_variant_comparison(
            result(), base_model="demo", reference_variant="Q6_K",
            performance_section="llm", case="2K", accuracy_section="mcq",
        )
    with pytest.raises(ValueError, match="unknown quality verdict"):
        build_variant_comparison(
            result(), base_model="demo", reference_variant="Q4_K_M",
            performance_section="llm", case="2K", accuracy_section="mcq",
            quality_verdicts={"Q8_0": "better"},
        )
