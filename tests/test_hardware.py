import pytest

import hardware
from models import LLM_MODELS


def test_parse_size_gb_gb_string():
    assert hardware.parse_size_gb("~4.9 GB") == 4.9


def test_parse_size_gb_mb_string():
    assert hardware.parse_size_gb("~274 MB") == pytest.approx(274 / 1024)


def test_parse_size_gb_malformed_returns_zero():
    assert hardware.parse_size_gb("who knows") == 0.0


# ── classify_gpu ──

def test_classify_gpu_amd_discrete():
    assert hardware.classify_gpu("AMD Radeon RX 7900 XTX") == "discrete"
    assert hardware.classify_gpu("AMD Radeon PRO W7900") == "discrete"
    assert hardware.classify_gpu("AMD Instinct MI300X") == "discrete"


def test_classify_gpu_amd_integrated():
    assert hardware.classify_gpu("AMD Radeon Graphics") == "integrated"
    assert hardware.classify_gpu("AMD Radeon 8060S Graphics") == "integrated"


def test_classify_gpu_intel_discrete():
    assert hardware.classify_gpu("Intel Arc B580") == "discrete"
    assert hardware.classify_gpu("Intel Arc A770") == "discrete"


def test_classify_gpu_intel_integrated():
    assert hardware.classify_gpu("Intel Arc Graphics") == "integrated"


# ── compute_memory_ceiling_gb ──

def test_ceiling_nvidia_uses_vram_minus_reserve():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Windows", total_ram_gb=32, gpu_vendor="nvidia", vram_gb=24)
    assert ceiling == pytest.approx(24 - hardware.VRAM_RESERVE_GB)
    assert "VRAM" in note


def test_ceiling_darwin_uses_ram_minus_reserve():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Darwin", total_ram_gb=16, gpu_vendor="integrated", vram_gb=None)
    assert ceiling == pytest.approx(16 - hardware.RAM_RESERVE_GB)
    assert "system RAM" in note


def test_ceiling_integrated_gpu_uses_ram_not_vram():
    """Integrated GPUs are treated like unified memory regardless of OS —
    even if a vram_gb figure were somehow passed in, the ceiling is total
    system RAM, not that number."""
    ceiling, _ = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=64, gpu_vendor="integrated", vram_gb=None)
    assert ceiling == pytest.approx(64 - hardware.RAM_RESERVE_GB)


def test_ceiling_ram_undetermined_returns_none():
    """RAM detection itself can fail (e.g. unparsable /proc/meminfo) — that
    must come back None rather than silently computing a bogus negative
    ceiling from a 0.0 fallback."""
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=None, gpu_vendor="integrated", vram_gb=None)
    assert ceiling is None
    assert "RAM" in note


def test_ceiling_no_gpu_uses_ram():
    ceiling, _ = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=32, gpu_vendor="none", vram_gb=None)
    assert ceiling == pytest.approx(32 - hardware.RAM_RESERVE_GB)


def test_ceiling_discrete_amd_with_known_vram():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=32, gpu_vendor="amd", vram_gb=16)
    assert ceiling == pytest.approx(16 - hardware.VRAM_RESERVE_GB)
    assert "VRAM" in note


def test_ceiling_discrete_amd_unknown_vram_returns_none():
    """No driver-agnostic VRAM query exists for discrete AMD/Intel on
    Windows — the ceiling must come back None (don't filter) rather than
    silently falling back to system RAM, which would be wrong for a
    discrete GPU."""
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Windows", total_ram_gb=32, gpu_vendor="amd", vram_gb=None)
    assert ceiling is None
    assert "manually" in note


def test_ceiling_discrete_intel_unknown_vram_returns_none():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Windows", total_ram_gb=32, gpu_vendor="intel", vram_gb=None)
    assert ceiling is None
    assert "manually" in note


# ── model_fits ──

def test_model_fits_true_when_well_under_ceiling():
    assert hardware.model_fits("~5.0 GB", 20.0) is True


def test_model_fits_false_when_over_ceiling():
    assert hardware.model_fits("~20.0 GB", 10.0) is False


def test_model_fits_none_when_ceiling_unknown():
    assert hardware.model_fits("~5.0 GB", None) is None


def test_model_fits_accounts_for_overhead_multiplier():
    # 10 GB model * 1.2 overhead = 12 GB required — fits a 15 GB ceiling,
    # not an 11 GB one.
    assert hardware.model_fits("~10.0 GB", 15.0) is True
    assert hardware.model_fits("~10.0 GB", 11.0) is False


def test_model_fits_against_real_catalog_values():
    """Sanity check against models.py's actual download_size values, not
    just synthetic strings — catches drift if MEMORY_OVERHEAD_MULTIPLIER or
    a model's download_size changes in a way that flips a real model's
    fit/no-fit outcome unexpectedly."""
    xsmall = next(m for m in LLM_MODELS if m["short"] == "llama3.2-3b-q4")
    assert hardware.model_fits(xsmall["download_size"], 8.0) is True

    large = next(m for m in LLM_MODELS if m["short"] == "deepseek-r1-70b")
    assert hardware.model_fits(large["download_size"], 8.0) is False
