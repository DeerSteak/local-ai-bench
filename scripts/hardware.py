"""Pure logic for deciding whether a model fits in available memory, used by
setup_check.py's picker — see docs/setup.md#memory-fit-estimate."""

import re

MEMORY_OVERHEAD_MULTIPLIER = 1.2
VRAM_RESERVE_GB = 1.0
RAM_RESERVE_GB  = 8.0


def parse_size_gb(s: str) -> float:
    """Parse a size string like '~4.9 GB' or '~274 MB' to float GB, 0.0 if unparsable."""
    s = s.strip().lstrip("~≈")
    try:
        if "MB" in s:
            return float(s.replace("MB", "").strip()) / 1024
        if "GB" in s:
            return float(s.replace("GB", "").strip())
    except ValueError:
        pass
    return 0.0


# See docs/setup.md#memory-fit-estimate for the classification heuristic and its default-to-integrated failure mode.
_AMD_DISCRETE_PATTERN = re.compile(r"(?:\bRX(?=\b|\d)|\bPRO\b|\bINSTINCT\b)")
_INTEL_DISCRETE_PATTERN = re.compile(r"\b[AB]\d{3}\b")


def classify_gpu(name: str) -> str:
    """Best-effort 'discrete' vs 'integrated' classification from a GPU name."""
    upper = name.upper()
    if _AMD_DISCRETE_PATTERN.search(upper):
        return "discrete"
    if _INTEL_DISCRETE_PATTERN.search(upper):
        return "discrete"
    return "integrated"


def rocminfo_gpu_names(output: str) -> list[str]:
    """Marketing Name values from rocminfo GPU agent blocks only — Agent 1 is
    normally the host CPU, so the first Marketing Name can be the processor."""
    names = []
    agent_blocks = re.split(r"(?m)^\s*Agent\s+\d+\s*$", output)[1:]
    for block in agent_blocks:
        device_type = re.search(r"(?m)^\s*Device Type:\s*(\S+)", block)
        marketing_name = re.search(r"(?m)^\s*Marketing Name:\s*(.+?)\s*$", block)
        if (device_type and device_type.group(1).upper() == "GPU"
                and marketing_name and marketing_name.group(1).strip()):
            names.append(marketing_name.group(1).strip())
    return names


def parse_nvidia_max_cuda_version(nvidia_smi_output: str) -> str | None:
    """Max CUDA version the driver supports, from plain `nvidia-smi`'s text
    header — see docs/setup.md's Windows (NVIDIA) note."""
    m = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*([\d.]+)", nvidia_smi_output)
    return m.group(1) if m else None


_CUDA_BIN_RE    = re.compile(r"^llama-.*-bin-win-cuda-([\d.]+)-x64\.zip$", re.IGNORECASE)
_CUDA_CUDART_RE = re.compile(r"^cudart-llama-bin-win-cuda-([\d.]+)-x64\.zip$", re.IGNORECASE)


def select_cuda_release_assets(assets: list[dict], max_cuda_version: str | None
                                ) -> tuple[dict, dict, str] | None:
    """Highest win-cuda-X.Y binary/cudart pair the driver supports, or None —
    see docs/setup.md's Windows (NVIDIA) note."""
    if not max_cuda_version:
        return None

    def _ver(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split("."))

    try:
        max_ver = _ver(max_cuda_version)
    except ValueError:
        return None

    by_version: dict[str, dict[str, dict]] = {}
    for asset in assets:
        name = asset.get("name", "")
        m = _CUDA_BIN_RE.match(name)
        if m:
            by_version.setdefault(m.group(1), {})["bin"] = asset
            continue
        m = _CUDA_CUDART_RE.match(name)
        if m:
            by_version.setdefault(m.group(1), {})["cudart"] = asset

    candidates = []
    for version, pair in by_version.items():
        if "bin" not in pair or "cudart" not in pair:
            continue
        try:
            v = _ver(version)
        except ValueError:
            continue
        if v <= max_ver:
            candidates.append((v, pair))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    version, pair = candidates[0][0], candidates[0][1]
    return pair["bin"], pair["cudart"], ".".join(str(p) for p in version)


def compute_memory_ceiling_gb(*, os_name: str, total_ram_gb: float | None,
                               gpu_vendor: str, vram_gb: float | None = None
                               ) -> tuple[float | None, str]:
    """Memory ceiling for this machine — see docs/setup.md#memory-fit-estimate.
    gpu_vendor "amd"/"intel" always means *discrete*; call classify_gpu() first."""
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
    """Estimated memory footprint: weights plus MEMORY_OVERHEAD_MULTIPLIER."""
    return parse_size_gb(download_size) * MEMORY_OVERHEAD_MULTIPLIER


def model_fits(download_size: str, ceiling_gb: float | None) -> bool | None:
    """True/False if ceiling_gb is known; None (meaning "unknown, don't
    filter on this") when ceiling_gb is None."""
    if ceiling_gb is None:
        return None
    return model_memory_requirement_gb(download_size) <= ceiling_gb


# Rounded UP to the next 0.1 GB — see docs/workloads.md.
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

# Encoder files each image model's "short" name needs alongside its checkpoint. SD1.5/SDXL bundle their own.
IMAGE_ENCODER_GROUPS = {
    "sd35-large": ("t5xxl_fp16.safetensors", "clip_l.safetensors", "clip_g.safetensors"),
    "flux-dev":   ("t5xxl_fp16.safetensors", "clip_l.safetensors", "ae.safetensors"),
    "flux2-dev":  ("mistral_3_small_flux2_fp8.safetensors", "flux2-vae.safetensors"),
}


def image_model_memory_requirement_gb(checkpoint: str, short: str) -> float:
    """Estimated memory footprint: checkpoint plus its IMAGE_ENCODER_GROUPS encoders, times MEMORY_OVERHEAD_MULTIPLIER."""
    weights_gb = CHECKPOINT_SIZES_GB.get(checkpoint, 0.0) + sum(
        ENCODER_SIZES_GB[f] for f in IMAGE_ENCODER_GROUPS.get(short, ()))
    return weights_gb * MEMORY_OVERHEAD_MULTIPLIER


def image_model_fits(checkpoint: str, short: str, ceiling_gb: float | None) -> bool | None:
    """True/False if ceiling_gb is known; None ("unknown, don't filter")
    otherwise — same contract as model_fits(), for image models."""
    if ceiling_gb is None:
        return None
    return image_model_memory_requirement_gb(checkpoint, short) <= ceiling_gb
