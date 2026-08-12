import pytest

from scripts.workloads.code_benchmark import CodeBenchmark
from scripts.workloads.code_sandbox import MAX_OUTPUT_BYTES, validate_candidate_code


@pytest.mark.parametrize(("code", "reason"), [
    ("import os", "imports are not allowed"),
    ("from socket import socket", "imports are not allowed"),
    ("open('outside', 'w')", "restricted name is not allowed: open"),
    ("getattr(object, 'x')", "restricted name is not allowed: getattr"),
    ("x.__class__", "dunder attribute access is not allowed"),
])
def test_static_policy_rejects_direct_escape_surfaces(code, reason):
    assert validate_candidate_code(code) == reason


def test_restricted_candidate_cannot_create_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = "def f():\n    open('escaped.txt', 'w').write('x')"
    result = CodeBenchmark.execute_tests(code, "f", [{"args": [], "expected": None}])
    assert result[0]["error"].startswith("unsafe generated code")
    assert not (tmp_path / "escaped.txt").exists()


def test_memory_exhaustion_is_stopped_by_parent_monitor():
    code = "def f():\n    return [0] * 100_000_000"
    result = CodeBenchmark.execute_tests(code, "f", [{"args": [], "expected": []}], timeout=3)
    assert result == [{"passed": False, "got": None, "error": "memory limit"}]


def test_single_huge_allocation_is_kernel_bounded():
    code = "def f():\n    return b'x' * 10_000_000_000"
    result = CodeBenchmark.execute_tests(code, "f", [{"args": [], "expected": b""}], timeout=3)
    assert result == [{"passed": False, "got": None, "error": "memory limit"}]


def test_stateful_memory_exhaustion_reports_memory_limit():
    code = "class C:\n    def allocate(self):\n        return b'x' * 10_000_000_000"
    tests = [{"init": [], "ops": [["allocate", []]], "expected": [b""]}]
    result = CodeBenchmark.execute_stateful_tests(code, "C", tests, timeout=3)
    assert result == [{"passed": False, "got": None, "error": "memory limit"}]


def test_candidate_output_is_bounded_and_cannot_fill_parent_memory():
    code = f"def f():\n    print('x' * {MAX_OUTPUT_BYTES * 2})\n    return 1"
    result = CodeBenchmark.execute_tests(code, "f", [{"args": [], "expected": 1}], timeout=3)
    assert result[0]["passed"] is False
    assert result == [{"passed": False, "got": None, "error": "output limit"}]
