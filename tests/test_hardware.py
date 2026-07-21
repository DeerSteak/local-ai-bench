import pytest

import hardware
from models import LLM_MODELS, IMAGE_MODELS


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
    assert hardware.classify_gpu("AMD Ryzen 9 7950X 16-Core Processor") == "integrated"


def test_classify_gpu_intel_discrete():
    assert hardware.classify_gpu("Intel Arc B580") == "discrete"
    assert hardware.classify_gpu("Intel Arc A770") == "discrete"


def test_classify_gpu_intel_integrated():
    assert hardware.classify_gpu("Intel Arc Graphics") == "integrated"


# ── rocminfo parsing ──

def test_rocminfo_gpu_names_ignores_cpu_agent_and_returns_gpu():
    output = """
*******
Agent 1
*******
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Device Type:             GPU
"""
    assert hardware.rocminfo_gpu_names(output) == ["AMD Radeon Graphics"]


def test_rocminfo_gpu_names_returns_all_gpu_agents():
    output = """
Agent 1
  Marketing Name: AMD Ryzen Processor
  Device Type: CPU
Agent 2
  Marketing Name: AMD Radeon Graphics
  Device Type: GPU
Agent 3
  Marketing Name: AMD Radeon RX 7900 XTX
  Device Type: GPU
"""
    assert hardware.rocminfo_gpu_names(output) == [
        "AMD Radeon Graphics",
        "AMD Radeon RX 7900 XTX",
    ]


def test_rocminfo_gpu_names_requires_an_agent_block_and_device_type():
    assert hardware.rocminfo_gpu_names("Marketing Name: AMD Radeon RX 7900 XTX") == []


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

    large = next(m for m in LLM_MODELS if m["short"] == "nemotron3-super-120b")
    assert hardware.model_fits(large["download_size"], 8.0) is False


# ── image_model_memory_requirement_gb / image_model_fits ──

def test_image_model_memory_requirement_checkpoint_only():
    # SDXL has no entry in IMAGE_ENCODER_GROUPS — checkpoint weight only.
    expected = hardware.CHECKPOINT_SIZES_GB["sd_xl_base_1.0.safetensors"] * hardware.MEMORY_OVERHEAD_MULTIPLIER
    assert hardware.image_model_memory_requirement_gb(
        "sd_xl_base_1.0.safetensors", "sdxl") == pytest.approx(expected)


def test_image_model_memory_requirement_includes_encoders():
    # Flux.1-dev needs its checkpoint plus t5xxl + clip_l + ae encoders.
    checkpoint_gb = hardware.CHECKPOINT_SIZES_GB["flux1-dev.safetensors"]
    encoder_gb = (hardware.ENCODER_SIZES_GB["t5xxl_fp16.safetensors"]
                  + hardware.ENCODER_SIZES_GB["clip_l.safetensors"]
                  + hardware.ENCODER_SIZES_GB["ae.safetensors"])
    expected = (checkpoint_gb + encoder_gb) * hardware.MEMORY_OVERHEAD_MULTIPLIER
    assert hardware.image_model_memory_requirement_gb(
        "flux1-dev.safetensors", "flux-dev") == pytest.approx(expected)


def test_image_model_fits_none_when_ceiling_unknown():
    assert hardware.image_model_fits("flux2-dev.safetensors", "flux2-dev", None) is None


def test_image_model_fits_false_when_encoders_push_it_over():
    """A checkpoint alone might fit, but Flux.2-dev's Mistral text encoder
    (~18 GB) is large enough that omitting it from the requirement — the bug
    this test guards against — would wrongly say it fits a 24 GB machine."""
    assert hardware.image_model_fits("flux2-dev.safetensors", "flux2-dev", 24.0) is False


def test_image_model_fits_true_on_large_ceiling():
    assert hardware.image_model_fits("v1-5-pruned-emaonly.safetensors", "sd15", 24.0) is True


def test_image_model_fits_against_real_catalog_values():
    """Sanity check against models.py's actual IMAGE_MODELS entries, not
    just synthetic checkpoint/short strings."""
    sd15 = next(m for m in IMAGE_MODELS if m["short"] == "sd15")
    assert hardware.image_model_fits(sd15["checkpoint"], sd15["short"], 24.0) is True

    flux2 = next(m for m in IMAGE_MODELS if m["short"] == "flux2-dev")
    assert hardware.image_model_fits(flux2["checkpoint"], flux2["short"], 24.0) is False
