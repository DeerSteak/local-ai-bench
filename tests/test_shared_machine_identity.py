from scripts.runtime.shared import _machine_identity, _nvidia_gpu_summary


def test_nvidia_summary_uses_first_name_and_aggregate_vram():
    output = "\n".join([
        "NVIDIA GeForce RTX 5060 Ti, 16384 MiB, 610.74",
        "NVIDIA GeForce RTX 5060 Ti, 16384 MiB, 610.74",
    ])
    assert _nvidia_gpu_summary(output) == ("NVIDIA GeForce RTX 5060 Ti", 32.0)


def test_nvidia_summary_does_not_report_partial_total():
    output = "\n".join([
        "NVIDIA GPU, 16384 MiB, 610.74",
        "NVIDIA GB10, N/A, 610.74",
    ])
    assert _nvidia_gpu_summary(output) == ("NVIDIA GPU", None)


def test_machine_identity_labels_ram_and_vram():
    assert _machine_identity("Intel CPU", "NVIDIA GPU", 64, 32) == (
        "Intel CPU / 64 GB RAM\nNVIDIA GPU / 32 GB VRAM"
    )


def test_machine_identity_keeps_gpu_when_vram_is_unknown():
    assert _machine_identity("Intel CPU", "Unknown GPU", 64) == (
        "Intel CPU / 64 GB RAM\nUnknown GPU"
    )
