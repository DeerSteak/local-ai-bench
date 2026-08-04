"""Run the fixed release security scan set and preserve machine-readable evidence."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.release.security_gate import security_gate_result


def scan_commands(repo_root, evidence_dir, python_executable=sys.executable):
    """Build the reviewed scanner commands without accepting caller-defined execution."""
    root = Path(repo_root).resolve()
    evidence = Path(evidence_dir).resolve()
    return (
        ("python_runtime_dependencies", [python_executable, "-m", "pip_audit", "-r",
         str(root / "requirements.txt"), "--format", "json", "--output",
         str(evidence / "python-runtime-audit.json")], root, evidence / "python-runtime-audit.json"),
        ("python_test_dependencies", [python_executable, "-m", "pip_audit", "-r",
         str(root / "tests" / "requirements.txt"), "--format", "json", "--output",
         str(evidence / "python-test-audit.json")], root, evidence / "python-test-audit.json"),
        ("python_static_analysis", [python_executable, "-m", "bandit", "-r",
         str(root / "scripts"), "-f", "json", "-o",
         str(evidence / "python-bandit.json")], root, evidence / "python-bandit.json"),
        ("npm_dependency_audit", ["npm", "audit", "--package-lock-only", "--json"],
         root / "dashboard", evidence / "npm-audit.json"),
    )


def run_release_scans(repo_root, staging_dir, evidence_dir, runner=subprocess.run):
    """Run every required scan and return a failing aggregate if any scan cannot pass."""
    evidence = Path(evidence_dir).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    artifact = security_gate_result(staging_dir)
    (evidence / "artifact-security-gate.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    checks = [{"name": "artifact_and_secret_scan", "passed": artifact["passed"],
               "evidence": "artifact-security-gate.json"}]
    for name, command, cwd, output_path in scan_commands(repo_root, evidence):
        try:
            completed = runner(command, cwd=cwd, capture_output=True, text=True, check=False)
            if not output_path.exists():
                output_path.write_text(completed.stdout or "", encoding="utf-8")
            checks.append({
                "name": name, "passed": completed.returncode == 0,
                "return_code": completed.returncode, "evidence": output_path.name,
            })
        except OSError as exc:
            checks.append({
                "name": name, "passed": False, "return_code": None,
                "error": type(exc).__name__, "evidence": None,
            })
    result = {"schema_version": 1, "passed": all(item["passed"] for item in checks),
              "checks": checks}
    (evidence / "release-scans.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python -m scripts.release.release_scans REPOSITORY STAGING_DIR EVIDENCE_DIR"
        )
    report = run_release_scans(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)
