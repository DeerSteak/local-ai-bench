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

# Weights + ~20% for KV-cache/runtime overhead. An approximation, not a
# precise per-context-length calculation — deliberately simple, same spirit
# as the disk-space check's flat buffers below.
MEMORY_OVERHEAD_MULTIPLIER = 1.2

# Headroom reserved on top of a model's estimated footprint before comparing
# against what the machine has, so "fits" doesn't mean "exactly to the byte."
VRAM_RESERVE_GB = 1.0   # driver / other GPU processes
RAM_RESERVE_GB  = 8.0   # OS + Ollama + everything else — flat buffer, same
                         # style as the existing disk-space check's 10 GB
                         # buffer. Applies to the shared-memory path (Apple
                         # Silicon unified memory, integrated GPUs, CPU-only)
                         # where the model competes with the OS for the same
                         # pool of RAM, unlike dedicated VRAM.


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
_AMD_DISCRETE_MARKERS = ("RX", "PRO", "INSTINCT")

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
    if any(marker in upper for marker in _AMD_DISCRETE_MARKERS):
        return "discrete"
    if _INTEL_DISCRETE_PATTERN.search(upper):
        return "discrete"
    return "integrated"


def compute_memory_ceiling_gb(*, os_name: str, total_ram_gb: float | None,
                               gpu_vendor: str, vram_gb: float | None = None
                               ) -> tuple[float | None, str]:
    """Decide how much memory a model can realistically use on this machine.

    gpu_vendor is one of "nvidia", "amd", "intel", "integrated", or "none".
    "amd"/"intel" here always means *discrete* — call classify_gpu() first
    and pass "integrated" instead when it returns that, so this function
    doesn't need to re-derive discrete/integrated itself.

    Returns (ceiling_gb, note). ceiling_gb is None when it can't be reliably
    determined (a discrete AMD/Intel GPU with no vram_gb — no
    driver-agnostic VRAM query is implemented for that path yet); callers
    should treat None as "don't filter, tell the user why" rather than
    blocking anything.
    """
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
