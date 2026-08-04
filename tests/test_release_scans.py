import json
from pathlib import Path
from subprocess import CompletedProcess

from release_scans import run_release_scans, scan_commands


def make_repo(tmp_path):
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "dashboard").mkdir()
    (root / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (root / "tests" / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    return root


def test_scan_commands_are_fixed_and_cover_dependency_and_static_scanners(tmp_path):
    root = make_repo(tmp_path)
    commands = scan_commands(root, tmp_path / "evidence", "/python")
    assert [name for name, _, _, _ in commands] == [
        "python_runtime_dependencies", "python_test_dependencies",
        "python_static_analysis", "npm_dependency_audit",
    ]
    assert commands[0][1][:3] == ["/python", "-m", "pip_audit"]
    assert commands[2][1][:3] == ["/python", "-m", "bandit"]
    assert commands[3][1] == ["npm", "audit", "--package-lock-only", "--json"]


def test_release_scans_preserve_evidence_and_pass_only_when_every_check_passes(tmp_path):
    root = make_repo(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "app.txt").write_text("safe", encoding="utf-8")
    evidence = tmp_path / "evidence"

    def runner(command, **kwargs):
        return CompletedProcess(command, 0, stdout='{"ok":true}', stderr="")

    result = run_release_scans(root, staging, evidence, runner)
    assert result["passed"] is True
    assert len(result["checks"]) == 5
    assert json.loads((evidence / "release-scans.json").read_text())["passed"] is True
    assert json.loads((evidence / "artifact-security-gate.json").read_text())["passed"] is True


def test_release_scans_fail_closed_for_findings_and_missing_tools(tmp_path):
    root = make_repo(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "hf.txt").write_text("not printed", encoding="utf-8")
    calls = [0]

    def runner(command, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise FileNotFoundError("scanner missing")
        return CompletedProcess(command, 1, stdout='{"findings":1}', stderr="secret detail")

    result = run_release_scans(root, staging, tmp_path / "evidence", runner)
    assert result["passed"] is False
    assert result["checks"][0]["passed"] is False
    assert result["checks"][1]["error"] == "FileNotFoundError"
    assert "secret detail" not in json.dumps(result)
