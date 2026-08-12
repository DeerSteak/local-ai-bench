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
