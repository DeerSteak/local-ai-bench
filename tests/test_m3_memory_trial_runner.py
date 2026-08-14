import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_memory_trial_runner_dry_run_alternates_pair_order(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "run_m3_memory_trials.sh"), "--model", "example:model",
         "--pairs", "2", "--wait", "0", "--out-dir", str(tmp_path), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    commands = [line for line in result.stdout.splitlines() if "run_bench.sh" in line]
    assert ["--memory-telemetry" in line for line in commands] == [False, True, True, False]
    assert all("--max-prompt-tokens 2048" in line for line in commands)
    assert not (tmp_path / "manifest.json").exists()


def test_memory_trial_runner_requires_a_model():
    result = subprocess.run(
        ["bash", str(ROOT / "run_m3_memory_trials.sh"), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "--model is required" in result.stderr
