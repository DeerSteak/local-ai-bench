import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_memory_trial_runner_dry_run_alternates_pair_order(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "run_telemetry_trials.sh"), "--model", "example:model",
         "--pairs", "2", "--wait", "0", "--out-dir", str(tmp_path), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    commands = [line for line in result.stdout.splitlines() if "run_bench.sh" in line]
    assert ["--memory-telemetry" in line for line in commands] == [False, True, True, False]
    assert ["--no-memory-telemetry" in line for line in commands] == [True, False, False, True]
    assert all("--max-prompt-tokens 2048" in line for line in commands)
    assert not (tmp_path / "manifest.json").exists()


def test_memory_trial_runner_requires_a_model():
    result = subprocess.run(
        ["bash", str(ROOT / "run_telemetry_trials.sh"), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "--model is required" in result.stderr


def test_power_trial_dry_run_compares_memory_only_with_combined_sampler(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "run_telemetry_trials.sh"), "--model", "example:model",
         "--telemetry", "power", "--pairs", "2", "--wait", "0",
         "--out-dir", str(tmp_path), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    commands = [line for line in result.stdout.splitlines() if "run_bench.sh" in line]
    assert ["--power-telemetry" in line for line in commands] == [False, True, True, False]
    assert all("--memory-telemetry" in line for line in commands)
    assert "sudo" not in result.stdout.lower()


def test_m5_pro_power_wrapper_previews_all_120_invocations(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "run_power_qualification_m5_pro.sh"),
         "--out-root", str(tmp_path), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    commands = [line for line in result.stdout.splitlines() if "run_bench.sh" in line]
    assert len(commands) == 120
    assert sum("0.25" in line for line in commands) == 40
    assert sum("0.5" in line for line in commands) == 40
    assert sum("1.0" in line for line in commands) == 40
    assert all("gemma3:1b-it-q4_K_M" in line for line in commands)


def test_trial_runner_rejects_unknown_telemetry_mode():
    result = subprocess.run(
        ["bash", str(ROOT / "run_telemetry_trials.sh"), "--model", "example:model",
         "--telemetry", "temperature", "--dry-run"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "must be memory or power" in result.stderr


def test_sustained_linux_wrapper_previews_repeated_aligned_soaks(tmp_path):
    output = tmp_path / "evidence"
    result = subprocess.run(
        ["bash", str(ROOT / "run_sustained_qualification_linux.sh"),
         "--model", "example:model", "--ambient-temp-c", "20.5",
         "--duration", "300", "--repeats", "2", "--out-dir", str(output),
         "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    commands = [line for line in result.stdout.splitlines() if "run_bench.sh" in line]
    assert len(commands) == 2
    assert all("--tests sustained" in line for line in commands)
    assert all("--memory-telemetry --power-telemetry" in line for line in commands)
    assert all("--sustained-duration 300" in line for line in commands)
    assert all("--ambient-temp-c 20.5" in line for line in commands)
    assert not output.exists()


def test_sustained_linux_wrapper_requires_qualification_inputs():
    result = subprocess.run(
        ["bash", str(ROOT / "run_sustained_qualification_linux.sh"), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "--model and --ambient-temp-c are required" in result.stderr


def test_sustained_linux_wrapper_rejects_too_short_duration():
    result = subprocess.run(
        ["bash", str(ROOT / "run_sustained_qualification_linux.sh"),
         "--model", "example:model", "--ambient-temp-c", "20", "--duration", "119",
         "--dry-run"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "at least 120 seconds" in result.stderr


def test_memory_trial_runner_rejects_dirty_source_tree(tmp_path):
    script = tmp_path / "run_telemetry_trials.sh"
    script.write_bytes((ROOT / "run_telemetry_trials.sh").read_bytes())
    python = tmp_path / "bench-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(subprocess.check_output(["which", "python3"], text=True).strip()))
    (tmp_path / ".gitignore").write_text("bench-env/\nresults/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "run_telemetry_trials.sh", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "initial"], cwd=tmp_path, check=True,
    )
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script), "--model", "example:model", "--pairs", "1"],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "requires a clean Git worktree" in result.stderr
