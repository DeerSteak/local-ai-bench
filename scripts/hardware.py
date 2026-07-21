"""
hardware.py — pure logic for deciding whether a model will fit on this
machine's available memory, used by setup_check.py's model picker.

No external dependencies (same constraint as models.py): safe to import
before requirements.txt is installed. All subprocess/hardware-query I/O stays
in setup_check.py — everything here is a plain function of already-gathered
values, which is what makes it unit-testable (setup_check.py itself can't be
imported/run in tests, see AGENTS.md).
"""

import re

# Weights + ~20% for runtime overhead (KV-cache for LLMs, activations for
# image models). An approximation, not a precise per-context-length
# calculation — deliberately simple, same spirit as the disk-space check's
# flat buffers below.
MEMORY_OVERHEAD_MULTIPLIER = 1.2

# Headroom reserved on top of a model's estimated footprint before comparing
# against what the machine has, so "fits" doesn't mean "exactly to the byte."
VRAM_RESERVE_GB = 1.0   # driver / other GPU processes
RAM_RESERVE_GB  = 8.0   # OS + the inference server + everything else on the shared-memory path (unified/integrated/CPU-only)


def parse_size_gb(s: str) -> float:
    """Parse a size string like '~4.9 GB' or '~274 MB' to float GB. Returns
    0.0 for anything unparsable, same fallback as the disk-space check this
    was extracted from."""
    s = s.strip().lstrip("~≈")
    try:
        if "MB" in s:
            return float(s.replace("MB", "").strip()) / 1024
        if "GB" in s:
            return float(s.replace("GB", "").strip())
    except ValueError:
        pass
    return 0.0


# AMD discrete cards are branded with one of these tokens; APU/integrated
# graphics is typically bare ("AMD Radeon Graphics", "Radeon 8060S Graphics").
_AMD_DISCRETE_PATTERN = re.compile(r"(?:\bRX(?=\b|\d)|\bPRO\b|\bINSTINCT\b)")

# Intel discrete Arc cards carry a model number (A380/A750/A770, B570/B580);
# integrated Arc graphics (Meteor Lake/Lunar Lake and newer) is just
# "Intel Arc Graphics" with no number.
_INTEL_DISCRETE_PATTERN = re.compile(r"\b[AB]\d{3}\b")


def classify_gpu(name: str) -> str:
    """Best-effort 'discrete' vs 'integrated' classification from a GPU name
    string. Heuristic, not authoritative — like this project's existing
    Intel Arc detection, it's built from naming conventions rather than
    hardware this project's maintainers have all been able to test against.
    Unknown/ambiguous names default to 'integrated' (the more permissive
    failure mode: it means falling back to the system-RAM ceiling, not
    wrongly capping to a VRAM number that doesn't apply)."""
    upper = name.upper()
    if _AMD_DISCRETE_PATTERN.search(upper):
        return "discrete"
    if _INTEL_DISCRETE_PATTERN.search(upper):
        return "discrete"
    return "integrated"


def rocminfo_gpu_names(output: str) -> list[str]:
    """Return Marketing Name values only from rocminfo GPU agent blocks.

    rocminfo normally lists the host CPU as Agent 1, so selecting the first
    Marketing Name can classify the processor instead of the GPU."""
    names = []
    agent_blocks = re.split(r"(?m)^\s*Agent\s+\d+\s*$", output)[1:]
    for block in agent_blocks:
        device_type = re.search(r"(?m)^\s*Device Type:\s*(\S+)", block)
        marketing_name = re.search(r"(?m)^\s*Marketing Name:\s*(.+?)\s*$", block)
        if (device_type and device_type.group(1).upper() == "GPU"
                and marketing_name and marketing_name.group(1).strip()):
            names.append(marketing_name.group(1).strip())
    return names


def compute_memory_ceiling_gb(*, os_name: str, total_ram_gb: float | None,
                               gpu_vendor: str, vram_gb: float | None = None
                               ) -> tuple[float | None, str]:
    """Decide how much memory a model can realistically use on this machine.
    gpu_vendor: "nvidia"/"amd"/"intel"/"integrated"/"none" — "amd"/"intel"
    always means *discrete* here, so call classify_gpu() first and pass
    "integrated" when it returns that. Returns (ceiling_gb, note); ceiling_gb
    is None when it can't be reliably determined — callers should treat that
    as "don't filter, tell the user why" rather than blocking anything."""
    if gpu_vendor == "nvidia" and vram_gb is not None:
        ceiling = vram_gb - VRAM_RESERVE_GB
        return ceiling, f"~{ceiling:.1f} GB (NVIDIA VRAM, minus {VRAM_RESERVE_GB:.0f} GB reserve)"

    if gpu_vendor in ("amd", "intel"):
        if vram_gb is not None:
            ceiling = vram_gb - VRAM_RESERVE_GB
            return ceiling, f"~{ceiling:.1f} GB ({gpu_vendor.upper()} VRAM, minus {VRAM_RESERVE_GB:.0f} GB reserve)"
        return None, (f"Couldn't determine VRAM for this {gpu_vendor.upper()} GPU — "
                       "models won't be filtered by memory. Check your GPU's VRAM manually.")

    # Darwin (unified memory), "integrated" GPU, or "none" (CPU-only) all
    # share the same ceiling: total system RAM is the only pool available.
    if total_ram_gb is None:
        return None, "Couldn't determine total system RAM — models won't be filtered by memory."
    ceiling = total_ram_gb - RAM_RESERVE_GB
    return ceiling, f"~{ceiling:.1f} GB (system RAM, minus {RAM_RESERVE_GB:.0f} GB reserve)"


def model_memory_requirement_gb(download_size: str) -> float:
    """Estimated memory footprint for a model given its models.py
    download_size string — weights plus overhead, see
    MEMORY_OVERHEAD_MULTIPLIER."""
    return parse_size_gb(download_size) * MEMORY_OVERHEAD_MULTIPLIER


def model_fits(download_size: str, ceiling_gb: float | None) -> bool | None:
    """True/False if ceiling_gb is known; None (meaning "unknown, don't
    filter on this") when ceiling_gb is None."""
    if ceiling_gb is None:
        return None
    return model_memory_requirement_gb(download_size) <= ceiling_gb


# On-disk size per checkpoint, rounded UP to the next 0.1 GB so the
# disk-space check errs toward requiring more space, not less.
CHECKPOINT_SIZES_GB = {
    "v1-5-pruned-emaonly.safetensors": 4.3,
    "sd_xl_base_1.0.safetensors": 7.0,
    "sd3.5_large.safetensors":    16.5,
    "flux1-dev.safetensors":      23.9,
    "flux2-dev.safetensors":      64.5,
}
ENCODER_SIZES_GB = {
    "t5xxl_fp16.safetensors":               9.8,
    "clip_l.safetensors":                   0.3,
    "clip_g.safetensors":                   1.4,
    "ae.safetensors":                       0.4,
    "flux2-vae.safetensors":                0.4,
    "mistral_3_small_flux2_fp8.safetensors": 18.1,
}

# Which encoder files (see ENCODER_SIZES_GB) each image model's "short" name
# needs loaded alongside its checkpoint at generation time. SD1.5/SDXL use
# none — their text encoder is bundled in the checkpoint file itself.
IMAGE_ENCODER_GROUPS = {
    "sd35-large": ("t5xxl_fp16.safetensors", "clip_l.safetensors", "clip_g.safetensors"),
    "flux-dev":   ("t5xxl_fp16.safetensors", "clip_l.safetensors", "ae.safetensors"),
    "flux2-dev":  ("mistral_3_small_flux2_fp8.safetensors", "flux2-vae.safetensors"),
}


def image_model_memory_requirement_gb(checkpoint: str, short: str) -> float:
    """Estimated memory footprint for an image model: its checkpoint plus
    whatever encoders it needs resident alongside it (see
    IMAGE_ENCODER_GROUPS), times the same overhead multiplier as LLM
    weights."""
    weights_gb = CHECKPOINT_SIZES_GB.get(checkpoint, 0.0) + sum(
        ENCODER_SIZES_GB[f] for f in IMAGE_ENCODER_GROUPS.get(short, ()))
    return weights_gb * MEMORY_OVERHEAD_MULTIPLIER


def image_model_fits(checkpoint: str, short: str, ceiling_gb: float | None) -> bool | None:
    """True/False if ceiling_gb is known; None ("unknown, don't filter")
    otherwise — same contract as model_fits(), for image models."""
    if ceiling_gb is None:
        return None
    return image_model_memory_requirement_gb(checkpoint, short) <= ceiling_gb
