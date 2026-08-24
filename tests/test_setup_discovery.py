from scripts.setup import setup_discovery


def _platform(monkeypatch, name):
    monkeypatch.setattr(setup_discovery.platform, "system", lambda: name)
    monkeypatch.setattr(setup_discovery.platform, "release", lambda: "release")
    monkeypatch.setattr(setup_discovery.platform, "machine", lambda: "machine")
    monkeypatch.setattr(setup_discovery.platform, "node", lambda: "node")


def test_discovers_linux_memory(monkeypatch, tmp_path):
    _platform(monkeypatch, "Linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       33554432 kB\n", encoding="utf-8")

    result = setup_discovery.discover_system(meminfo)

    assert result.total_ram_gb == 32
    assert result.chip is None


def test_discovers_macos_chip_and_memory(monkeypatch):
    _platform(monkeypatch, "Darwin")
    outputs = iter(["Apple M4 Max\n", f"{64 * 1024 ** 3}\n"])
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output", lambda *_args, **_kwargs: next(outputs),
    )

    result = setup_discovery.discover_system()

    assert result.chip == "Apple M4 Max"
    assert result.total_ram_gb == 64


def test_missing_memory_is_nonfatal(monkeypatch, tmp_path):
    _platform(monkeypatch, "Linux")

    result = setup_discovery.discover_system(tmp_path / "missing")

    assert result.total_ram_gb is None


def test_discovers_nvidia_inventory_and_capabilities(monkeypatch):
    outputs = iter([
        "RTX 5090, 32768 MiB, 600.1\n", "12.0\n", "CUDA Version: 13.0\n",
    ])
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output", lambda *_args, **_kwargs: next(outputs),
    )

    result = setup_discovery.discover_nvidia()

    assert result.available
    assert result.total_vram_gb == 32
    assert result.compute_capability == "12.0"
    assert result.max_cuda_version == "13.0"


def test_missing_nvidia_is_empty(monkeypatch):
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert not setup_discovery.discover_nvidia().available


def test_discovers_discrete_rocm_memory(monkeypatch):
    outputs = iter([
        "Agent 1\n  Name: gfx1100\n  Marketing Name: AMD Radeon RX 7900 XTX\n"
        "  Device Type: GPU\n",
        '{"card0": {"VRAM Total Memory (B)": "25769803776"}}',
    ])
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output", lambda *_args, **_kwargs: next(outputs),
    )

    result = setup_discovery.discover_rocm()

    assert result.available
    assert result.kind == "discrete"
    assert result.gfx_targets == ["gfx1100"]
    assert result.total_vram_gb == 24


def test_missing_rocm_is_empty(monkeypatch):
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert not setup_discovery.discover_rocm().available


def test_discovers_windows_amd_gpu(monkeypatch):
    _platform(monkeypatch, "Windows")
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: "AMD Radeon RX 7900 XTX\n",
    )

    result = setup_discovery.discover_windows_gpu()

    assert result.vendor == "amd"
    assert result.kind == "discrete"


def test_discovers_linux_intel_arc(monkeypatch):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: "00:02.0 VGA compatible controller: Intel Arc A770\n",
    )

    result = setup_discovery.discover_linux_intel_gpu()

    assert result.vendor == "intel"
    assert result.name == "Intel Arc A770"


def test_discovers_linux_arc_pro_b65_pci_codename(monkeypatch):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: (
            "18:00.0 VGA compatible controller: Intel Corporation "
            "Battlemage G31 [Intel Graphics] [8086:e222]\n"
        ),
    )

    result = setup_discovery.discover_linux_intel_gpu()

    assert result.vendor == "intel"
    assert result.kind == "discrete"
    assert result.name is not None
    assert "Battlemage G31" in result.name


def test_discovers_intel_vram_from_xpu_smi(monkeypatch, tmp_path):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(setup_discovery.shutil, "which", lambda _name: "/usr/bin/xpu-smi")
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: "| 1024 MiB / 24576 MiB | 20% Default |\n",
    )

    assert setup_discovery.discover_intel_vram_gb(sysfs_root=tmp_path) == 24


def test_discovers_intel_vram_from_linux_drm_driver(monkeypatch, tmp_path):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(setup_discovery.shutil, "which", lambda _name: None)
    intel = tmp_path / "card1" / "device"
    intel.mkdir(parents=True)
    (intel / "vendor").write_text("0x8086\n", encoding="utf-8")
    (intel / "mem_info_vram_total").write_text(str(24 * 1024 ** 3), encoding="utf-8")
    other = tmp_path / "card2" / "device"
    other.mkdir(parents=True)
    (other / "vendor").write_text("0x1002\n", encoding="utf-8")
    (other / "mem_info_vram_total").write_text(str(16 * 1024 ** 3), encoding="utf-8")

    assert setup_discovery.discover_intel_vram_gb(sysfs_root=tmp_path) == 24


def test_intel_vram_discovery_is_unknown_without_tool_or_driver_counter(monkeypatch, tmp_path):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(setup_discovery.shutil, "which", lambda _name: None)

    assert setup_discovery.discover_intel_vram_gb(sysfs_root=tmp_path) is None


def test_discovers_linux_amd_gpu_before_rocm_is_installed(monkeypatch):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: (
            "0d:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Navi 44 [Radeon RX 9060 XT] [1002:7550]\n"
        ),
    )

    result = setup_discovery.discover_linux_amd_gpu()

    assert result.vendor == "amd"
    assert result.kind == "discrete"
    assert result.name is not None and "Radeon RX 9060 XT" in result.name


def test_discovers_linux_nvidia_gpu_before_driver_is_installed(monkeypatch):
    _platform(monkeypatch, "Linux")
    monkeypatch.setattr(
        setup_discovery.subprocess, "check_output",
        lambda *_args, **_kwargs: (
            "0d:00.0 VGA compatible controller: NVIDIA Corporation GB202 "
            "[GeForce RTX 5090] [10de:2b85]\n"
        ),
    )

    result = setup_discovery.discover_linux_nvidia_gpu()

    assert result.vendor == "nvidia"
    assert result.kind == "discrete"
    assert result.name is not None and "RTX 5090" in result.name
