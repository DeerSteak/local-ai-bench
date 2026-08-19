from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_unix_launcher_runs_normal_setup_then_normal_benchmark_wrapper():
    text = (ROOT / "run_qualification.sh").read_text()
    assert 'setup.sh" --qualification "$ENGINE"' in text
    assert "scripts.release.qualification_run" in text
    assert "qualification-env" not in text
    assert "qualification_automation" not in text


def test_windows_launcher_runs_normal_setup_then_normal_benchmark_wrapper():
    text = (ROOT / "run_qualification.bat").read_text()
    assert "call setup.bat --qualification %ENGINE%" in text
    assert "scripts.release.qualification_run" in text
    assert "qualification-env" not in text
    assert "qualification_automation" not in text


def test_posix_qualification_launcher_has_valid_shell_syntax():
    result = subprocess.run(
        ["/bin/bash", "-n", ROOT / "run_qualification.sh"], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_target_listing_needs_no_python_environment():
    result = subprocess.run(
        [ROOT / "run_qualification.sh", "--list-targets"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "macos-m5-pro-llamacpp-metal",
        "geforce-windows-llamacpp-cuda",
        "radeon-windows-llamacpp-vulkan",
        "intel-arc-windows-llamacpp-vulkan",
        "geforce-wsl2-llamacpp-cuda",
        "geforce-wsl2-vllm-cuda",
        "radeon-wsl2-llamacpp-rocm",
        "radeon-wsl2-vllm-rocm",
        "nvidia-linux-llamacpp-cuda",
        "nvidia-linux-vllm-cuda",
        "ryzen-ai-halo-llamacpp-rocm",
        "ryzen-ai-halo-vllm-rocm",
        "dgx-spark-llamacpp-cuda",
        "dgx-spark-vllm-cuda",
    ]


def test_unknown_target_is_rejected_before_setup():
    result = subprocess.run(
        [ROOT / "run_qualification.sh", "invented-target"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Unknown qualification target" in result.stderr
