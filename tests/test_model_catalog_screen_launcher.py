from pathlib import Path
import shutil
import subprocess

from scripts.release.model_catalog_screen import load_source_audit


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_model_catalog_screens.sh"


def test_catalog_screen_launcher_has_valid_shell_syntax():
    result = subprocess.run(
        ["/bin/bash", "-n", LAUNCHER], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_catalog_screen_launcher_lists_every_source_ready_llm_engine_pair():
    result = subprocess.run(
        [LAUNCHER, "--list"], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    candidates = [
        item["id"] for item in load_source_audit()["candidates"]
        if item["family"] == "llm" and item["status"] == "source_ready"
    ]
    assert result.stdout.splitlines() == [
        f"{candidate}\t{engine}"
        for candidate in candidates for engine in ("llamacpp", "vllm")
    ]


def test_catalog_screen_launcher_can_filter_the_matrix_by_engine():
    result = subprocess.run(
        [LAUNCHER, "--list", "--engine", "vllm"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 4
    assert all(line.endswith("\tvllm") for line in result.stdout.splitlines())


def test_catalog_screen_launcher_can_skip_a_known_failed_cell():
    result = subprocess.run(
        [LAUNCHER, "--list", "--skip-cell", "qwen3.8-27b/llamacpp"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 7
    assert "qwen3.8-27b\tllamacpp" not in result.stdout.splitlines()
    assert "qwen3.8-27b\tvllm" in result.stdout.splitlines()


def test_catalog_screen_launcher_rejects_unknown_arguments_without_running():
    result = subprocess.run(
        [LAUNCHER, "--engine", "future"], cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr

    result = subprocess.run(
        [LAUNCHER, "--skip-cell", "unknown/llamacpp"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2


def test_catalog_screen_launcher_continues_matrix_and_reports_failed_cells(tmp_path):
    launcher = tmp_path / LAUNCHER.name
    shutil.copy2(LAUNCHER, launcher)
    python = tmp_path / "bench-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "case \"$*\" in *\"muse-glimmer-30b --engine vllm\"*) exit 7;; esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    call_log = tmp_path / "calls.log"
    result = subprocess.run(
        [launcher, "--output-root", str(tmp_path / "evidence")],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "CALL_LOG": str(call_log)},
    )
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 1
    assert len(calls) == 8
    assert "gemma-4-26b-a4b --engine vllm" in calls[-1]
    assert all(f"--output-root {tmp_path / 'evidence'}" in call for call in calls)
    assert "Catalog screen matrix complete: 7 passed, 1 failed." in result.stdout
    assert "FAILED: muse-glimmer-30b/vllm (exit 7)" in result.stderr
