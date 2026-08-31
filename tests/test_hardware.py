import pytest

from scripts.runtime import hardware
from scripts.workloads.models import LLM_MODELS, IMAGE_MODELS


def test_parse_size_gb_gb_string():
    assert hardware.parse_size_gb("~4.9 GB") == 4.9


def test_parse_size_gb_mb_string():
    assert hardware.parse_size_gb("~274 MB") == pytest.approx(274 / 1024)


def test_parse_size_gb_malformed_returns_zero():
    assert hardware.parse_size_gb("who knows") == 0.0


def test_nvidia_smi_resolves_the_wsl_driver_location_when_path_omits_it(tmp_path):
    executable = tmp_path / "nvidia-smi"
    executable.write_text("driver bridge")
    assert hardware.nvidia_smi_executable(
        which_fn=lambda _name: None, wsl_path=executable,
    ) == str(executable)


def test_nvidia_smi_prefers_the_normal_path_discovery(tmp_path):
    assert hardware.nvidia_smi_executable(
        which_fn=lambda _name: "/usr/bin/nvidia-smi", wsl_path=tmp_path / "other",
    ) == "/usr/bin/nvidia-smi"


def test_rocm_tool_resolves_opt_install_when_path_omits_it(tmp_path):
    executable = tmp_path / "rocminfo"
    executable.write_text("tool", encoding="utf-8")
    assert hardware.rocm_executable(
        "rocminfo", which_fn=lambda _name: None, rocm_bin=tmp_path,
    ) == str(executable)


def test_rocm_tool_prefers_path_and_returns_none_when_absent(tmp_path):
    assert hardware.rocm_executable(
        "rocminfo", which_fn=lambda _name: "/usr/bin/rocminfo", rocm_bin=tmp_path,
    ) == "/usr/bin/rocminfo"
    assert hardware.rocm_executable(
        "rocminfo", which_fn=lambda _name: None, rocm_bin=tmp_path,
    ) is None


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
    assert hardware.classify_gpu("Intel Corporation Battlemage G31 [Intel Graphics]") == "discrete"
    assert hardware.classify_gpu("Intel Corporation Device e222 [8086:e222]") == "discrete"


def test_intel_xpu_display_accepts_arc_and_battlemage_but_not_generic_intel():
    assert hardware.is_intel_xpu_display("Intel Arc Pro B65")
    assert hardware.is_intel_xpu_display("Intel Corporation Battlemage G31 [Intel Graphics]")
    assert hardware.is_intel_xpu_display("Intel Corporation Device e222 [8086:e222]")
    assert not hardware.is_intel_xpu_display("Intel UHD Graphics 770")


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


# ── parse_nvidia_max_cuda_version ──

def test_parse_nvidia_max_cuda_version_classic_header():
    output = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07   CUDA Version: 12.4         |
+-----------------------------------------------------------------------------------------+
"""
    assert hardware.parse_nvidia_max_cuda_version(output) == "12.4"


def test_parse_nvidia_max_cuda_version_umd_header():
    """Newer drivers report 'CUDA UMD Version' rather than 'CUDA Version'."""
    output = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.74                 KMD Version: 610.74        CUDA UMD Version: 13.3     |
+-----------------------------------------------------------------------------------------+
"""
    assert hardware.parse_nvidia_max_cuda_version(output) == "13.3"


def test_parse_nvidia_max_cuda_version_missing_returns_none():
    assert hardware.parse_nvidia_max_cuda_version("no nvidia here") is None


def test_parse_nvidia_gpus_preserves_each_device_capacity():
    output = "\n".join([
        "NVIDIA GeForce RTX 5060 Ti, 16384 MiB, 610.74",
        "NVIDIA GeForce RTX 5060 Ti, 16384 MiB, 610.74",
        "malformed",
    ])
    assert hardware.parse_nvidia_gpus(output) == [
        {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0, "driver": "610.74"},
        {"name": "NVIDIA GeForce RTX 5060 Ti", "vram_gb": 16.0, "driver": "610.74"},
    ]


def test_parse_rocm_smi_gpus_preserves_each_device_capacity():
    output = """{
      "card0": {"VRAM Total Memory (B)": "17179869184"},
      "card1": {"VRAM Total Memory (B)": 17179869184}
    }"""
    assert hardware.parse_rocm_smi_gpus(output, ["Radeon A", "Radeon B"]) == [
        {"name": "Radeon A", "vram_gb": 16.0, "vendor": "amd", "backend": "rocm"},
        {"name": "Radeon B", "vram_gb": 16.0, "vendor": "amd", "backend": "rocm"},
    ]
    assert hardware.parse_rocm_smi_gpus("not json", []) == []


def test_gpu_tensor_split_reduces_nominal_device_capacities():
    devices = [
        {"backend": "cuda", "vram_gb": 31.984375},
        {"backend": "cuda", "vram_gb": 15.921875},
    ]

    assert hardware.gpu_tensor_split(devices) == "2,1"
    assert hardware.gpu_tensor_split([
        {"backend": "cuda", "vram_gb": 24},
        {"backend": "cuda", "vram_gb": 16},
    ]) == "3,2"


@pytest.mark.parametrize("devices", [
    [],
    [{"backend": "vulkan", "vram_gb": 16}],
    [{"backend": "vulkan", "vram_gb": 16}, {"backend": "cuda", "vram_gb": 16}],
    [{"backend": "vulkan", "vram_gb": 16}, {"backend": "vulkan", "vram_gb": None}],
    [{"backend": "vulkan", "vram_gb": 16}, {"backend": "vulkan", "vram_gb": 1}],
])
def test_gpu_tensor_split_rejects_incomplete_or_mixed_topologies(devices):
    assert hardware.gpu_tensor_split(devices) is None


def test_gpu_device_selection_pins_vulkan_inventory_order_only():
    vulkan = [{"backend": "vulkan"}, {"backend": "vulkan"}]
    assert hardware.gpu_device_selection(vulkan) is None
    assert hardware.gpu_device_selection([{"backend": "cuda"}, {"backend": "cuda"}]) is None
    assert hardware.gpu_device_selection([{"backend": "vulkan"}]) is None


def test_vulkan_split_omits_unverified_device_indices_and_ratios():
    devices = [
        {"backend": "vulkan", "vram_gb": 16},
        {"backend": "vulkan", "vram_gb": 32},
    ]
    assert hardware.gpu_device_selection(devices) is None
    assert hardware.gpu_tensor_split(devices) is None


def test_vulkan_split_does_not_assume_discovery_matches_runtime_order():
    devices = [
        {"backend": "vulkan", "name": "AMD Radeon Graphics", "vram_gb": 64},
        {"backend": "vulkan", "name": "AMD Radeon RX 7600", "vram_gb": 8},
        {"backend": "vulkan", "name": "AMD Radeon PRO W7800", "vram_gb": 32},
    ]
    assert hardware.gpu_device_selection(devices) is None
    assert hardware.gpu_tensor_split(devices) is None


# ── select_cuda_release_assets ──

def _asset(name, size=100):
    return {"name": name, "size": size, "browser_download_url": f"https://example.com/{name}"}


REAL_SHAPED_ASSETS = [
    _asset("llama-b10106-bin-win-cuda-12.4-x64.zip"),
    _asset("cudart-llama-bin-win-cuda-12.4-x64.zip"),
    _asset("llama-b10106-bin-win-cuda-13.3-x64.zip"),
    _asset("cudart-llama-bin-win-cuda-13.3-x64.zip"),
    _asset("llama-b10106-bin-win-vulkan-x64.zip"),
    _asset("llama-b10106-bin-win-cpu-x64.zip"),
    _asset("llama-b10106-bin-win-hip-x64.zip"),
]


def test_select_cuda_release_assets_picks_exact_match():
    result = hardware.select_cuda_release_assets(REAL_SHAPED_ASSETS, "13.3")
    assert result is not None
    bin_asset, cudart_asset, version = result
    assert version == "13.3"
    assert bin_asset["name"] == "llama-b10106-bin-win-cuda-13.3-x64.zip"
    assert cudart_asset["name"] == "cudart-llama-bin-win-cuda-13.3-x64.zip"


def test_select_cuda_release_assets_picks_highest_not_exceeding_driver():
    """A driver that only supports up to 12.9 must not get the 13.3 build."""
    result = hardware.select_cuda_release_assets(REAL_SHAPED_ASSETS, "12.9")
    assert result is not None
    _, _, version = result
    assert version == "12.4"


def test_select_cuda_release_assets_none_when_driver_too_old():
    assert hardware.select_cuda_release_assets(REAL_SHAPED_ASSETS, "11.0") is None


def test_select_cuda_release_assets_none_when_no_driver_version():
    assert hardware.select_cuda_release_assets(REAL_SHAPED_ASSETS, None) is None


def test_select_cuda_release_assets_none_when_release_has_no_cuda_builds():
    vulkan_only = [_asset("llama-b10106-bin-win-vulkan-x64.zip")]
    assert hardware.select_cuda_release_assets(vulkan_only, "13.3") is None


def test_select_cuda_release_assets_requires_matching_cudart_pair():
    """A binary with no matching cudart runtime zip must not be selected —
    it would be missing DLLs it needs at runtime."""
    orphan_bin = [_asset("llama-b10106-bin-win-cuda-13.3-x64.zip")]
    assert hardware.select_cuda_release_assets(orphan_bin, "13.3") is None


def test_select_cuda_release_assets_ignores_unrelated_assets():
    assert hardware.select_cuda_release_assets(
        [_asset("llama-b10106-bin-win-cpu-x64.zip"), _asset("llama-b10106-bin-win-hip-x64.zip")],
        "13.3",
    ) is None


# ── compute_memory_ceiling_gb ──

def test_ceiling_nvidia_uses_vram_minus_reserve():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Windows", total_ram_gb=32, gpu_vendor="nvidia", vram_gb=24)
    assert ceiling == pytest.approx(24 - hardware.VRAM_RESERVE_GB)
    assert "VRAM" in note


def test_ceiling_multi_nvidia_reserves_memory_on_every_device():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Windows", total_ram_gb=64, gpu_vendor="nvidia",
        vram_gb=32, device_vram_gb=[16, 16],
    )
    assert ceiling == pytest.approx(30)
    assert "2 NVIDIA GPUs" in note


def test_ceiling_darwin_uses_ram_minus_reserve():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Darwin", total_ram_gb=16, gpu_vendor="integrated", vram_gb=None)
    assert ceiling == pytest.approx(16 - hardware.RAM_RESERVE_GB)
    assert "system RAM" in note


def test_ceiling_integrated_gpu_uses_ram_not_vram():
    """Integrated GPUs use total system RAM as ceiling even if vram_gb is passed."""
    ceiling, _ = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=64, gpu_vendor="integrated", vram_gb=None)
    assert ceiling == pytest.approx(64 - hardware.RAM_RESERVE_GB)


def test_ceiling_ram_undetermined_returns_none():
    """Failed RAM detection must return None, not a bogus negative ceiling."""
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


def test_ceiling_multi_amd_reserves_memory_on_every_device():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=64, gpu_vendor="amd",
        vram_gb=32, device_vram_gb=[16, 16],
    )
    assert ceiling == pytest.approx(30)
    assert "2 AMD GPUs" in note


def test_ceiling_discrete_amd_unknown_vram_returns_none():
    """Unknown discrete VRAM must return None, not fall back to system RAM."""
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
    """Sanity check against models.py's real download_size values, not synthetic strings."""
    xsmall = next(m for m in LLM_MODELS if m["short"] == "granite4.1-3b-q4")
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


def test_z_image_memory_requirement_includes_qwen_encoder_and_vae():
    weights = (hardware.CHECKPOINT_SIZES_GB["z_image_turbo_bf16.safetensors"]
               + hardware.ENCODER_SIZES_GB["qwen_3_4b.safetensors"]
               + hardware.ENCODER_SIZES_GB["z_image_ae.safetensors"])
    assert hardware.image_model_weights_gb(
        "z_image_turbo_bf16.safetensors", "z-image-turbo",
    ) == pytest.approx(weights)
    assert hardware.image_model_memory_requirement_gb(
        "z_image_turbo_bf16.safetensors", "z-image-turbo",
    ) == pytest.approx(weights * hardware.MEMORY_OVERHEAD_MULTIPLIER)


def test_image_model_fits_none_when_ceiling_unknown():
    assert hardware.image_model_fits("flux2-dev.safetensors", "flux2-dev", None) is None


def test_image_model_fits_false_when_encoders_push_it_over():
    """Regression guard: omitting Flux.2-dev's ~18GB Mistral encoder would wrongly say it fits 24 GB."""
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


def test_parse_rocm_version():
    assert hardware.parse_rocm_version("HIP version: 6.4.43483-a187df25c") == (6, 4)
    assert hardware.parse_rocm_version("6.3.0-63") == (6, 3)
    assert hardware.parse_rocm_version("unknown") is None
    assert hardware.parse_rocm_version("") is None
    assert hardware.parse_rocm_version(None) is None


STRIX_HALO_ROCMINFO = """
Agent 1
  Name:                    AMD Ryzen AI Max+ 395
  Marketing Name:          AMD Ryzen AI Max+ 395 w/ Radeon 8060S
  Device Type:             CPU
Agent 2
  Name:                    gfx1151
  Marketing Name:          AMD Radeon Graphics
  Device Type:             GPU
"""

MI300_ROCMINFO = """
Agent 1
  Name:                    AMD EPYC 9654
  Device Type:             CPU
Agent 2
  Name:                    gfx942:sramecc+:xnack-
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
Agent 3
  Name:                    gfx942:sramecc+:xnack-
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
"""


def test_rocminfo_gfx_targets_ignores_the_cpu_agent():
    assert hardware.rocminfo_gfx_targets(STRIX_HALO_ROCMINFO) == ["gfx1151"]


def test_rocminfo_gfx_targets_strips_feature_suffixes_and_deduplicates():
    assert hardware.rocminfo_gfx_targets(MI300_ROCMINFO) == ["gfx942"]


def test_rocminfo_gfx_targets_is_empty_without_gpu_agents():
    assert hardware.rocminfo_gfx_targets("") == []
    assert hardware.rocminfo_gfx_targets("Agent 1\n  Name: cpu\n  Device Type: CPU\n") == []


# ── unified-memory NVIDIA parts (GB10 / DGX Spark) ──

def test_a_gpu_reporting_no_vram_is_still_detected():
    """GB10 shares memory with the host, so nvidia-smi has no dedicated figure to give.
    Dropping the row made setup conclude there was no GPU at all."""
    devices = hardware.parse_nvidia_gpus("NVIDIA GB10, [N/A], 580.173.02")
    assert devices == [{"name": "NVIDIA GB10", "vram_gb": None, "driver": "580.173.02"}]


def test_vram_field_accepts_every_unit_nvidia_smi_uses():
    assert hardware.parse_nvidia_vram_gb("16384 MiB") == 16.0
    assert hardware.parse_nvidia_vram_gb("120 GiB") == 120.0
    assert hardware.parse_nvidia_vram_gb("16384MiB") == 16.0
    for junk in ("[N/A]", "N/A", "", None, "Insufficient Permissions"):
        assert hardware.parse_nvidia_vram_gb(junk) is None


def test_decimal_mb_and_gb_convert_using_the_same_1000_1024_ratio():
    """MB/GB are decimal (1000-based) units; converting to GiB-equivalent must divide
    by 1024**3, not 1024**2 — a factor-of-1024 slip understates VRAM by ~2.4%."""
    assert hardware.parse_nvidia_vram_gb("24000 MB") == pytest.approx(24000 * 1000**2 / 1024**3)
    assert hardware.parse_nvidia_vram_gb("24 GB") == pytest.approx(24 * 1000**3 / 1024**3)


def test_rows_without_a_name_are_still_rejected():
    assert hardware.parse_nvidia_gpus(", 16384 MiB, 610.74") == []
    assert hardware.parse_nvidia_gpus("malformed") == []


def test_unknown_vram_falls_back_to_the_system_ram_ceiling():
    ceiling, note = hardware.compute_memory_ceiling_gb(
        os_name="Linux", total_ram_gb=121.6, gpu_vendor="nvidia",
        vram_gb=None, device_vram_gb=None,
    )
    assert ceiling == pytest.approx(121.6 - hardware.RAM_RESERVE_GB)
    assert "system RAM" in note
