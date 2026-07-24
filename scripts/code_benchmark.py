"""code_benchmark.py — coding accuracy benchmark: each model answers every
problem in scripts/data/code_problems.json once at temperature 0, then that
answer is run against the problem's visible and hidden test cases in an
isolated subprocess; correct only if every test case passes. Scored overall
and per-category, same shape as MCQBenchmark/MathBenchmark.

Two problem shapes: function problems (model writes one function, tests are
args/expected pairs) and stateful problems (model writes a class, tests are
init/ops/expected sequences run against one fresh instance). visible_tests
are shown in the prompt as worked examples; hidden_tests are grading-only,
run alongside them so memorizing the visible cases isn't enough.
"""

import json
import math
import re
import secrets
import subprocess
import sys
from pathlib import Path

import config
from shared import Shared


class CodeBenchmark:
    CODE_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "code_problems.json"

    # Records models that crashed the engine's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    CODE_CRASH_CACHE = Path(".code_crash_cache.json")

    # -1 delegates the finite per-pass limits to chat's token_budget split.
    CODE_NUM_PREDICT = -1

    # Process-isolation timeout, not a security sandbox — just enough to survive bad generated code.
    CODE_EXEC_TIMEOUT = 5

    # Pulls the code out of a fenced block (```python ... ``` or ``` ... ```);
    # DOTALL so the body can span multiple lines.
    _FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

    @staticmethod
    def load_questions(path: Path = CODE_DATA_PATH) -> list[dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _format_function_example(function_name: str, test: dict) -> str:
        args_repr = ", ".join(repr(a) for a in test["args"])
        return f"{function_name}({args_repr}) == {test['expected']!r}"

    @staticmethod
    def _format_stateful_example(class_name: str, test: dict) -> str:
        init_repr = ", ".join(repr(a) for a in test.get("init", []))
        lines = [f"obj = {class_name}({init_repr})"]
        for (method, args), expected in zip(test["ops"], test["expected"]):
            args_repr = ", ".join(repr(a) for a in args)
            lines.append(f"obj.{method}({args_repr})  # -> {expected!r}")
        return "\n".join(lines)

    @staticmethod
    def _build_examples_block(question: dict) -> str:
        """Renders `visible_tests` as worked examples, or '' if there are
        none. `hidden_tests` are deliberately never passed to this method."""
        tests = question.get("visible_tests", [])
        if not tests:
            return ""
        if "class_name" in question:
            examples = [CodeBenchmark._format_stateful_example(question["class_name"], t) for t in tests]
        else:
            examples = [CodeBenchmark._format_function_example(question["function_name"], t) for t in tests]
        return "Examples:\n" + "\n\n".join(examples) + "\n\n"

    @staticmethod
    def build_prompt(question: dict) -> str:
        examples = CodeBenchmark._build_examples_block(question)
        if "class_name" in question:
            return (
                f"{question['prompt']}\n\n{examples}"
                f"Respond with only the class definition for {question['class_name']}, "
                "as a single Python code block, with no explanation."
            )
        return (
            f"{question['prompt']}\n\n{examples}"
            f"Respond with only the function definition for {question['function_name']}, "
            "as a single Python code block, with no explanation."
        )

    @staticmethod
    def extract_code(response_text: str) -> str:
        """Pull the model's code out of its free-form reply. Prefers the
        *last* fenced code block, not the first — a reasoning model's
        unbounded output often drafts/revises before its final answer. Falls
        back to the whole stripped reply if there's no fence at all."""
        if not response_text:
            return ""
        matches = list(CodeBenchmark._FENCE_RE.finditer(response_text))
        if matches:
            return matches[-1].group(1).strip()
        return response_text.strip()

    @staticmethod
    def _values_close(got, expected) -> bool:
        """Exact equality, except floats use math.isclose (1e-9) — a
        legitimately correct solution can differ from a stored float
        `expected` in the last bit from a different but valid order of
        operations. Recurses into lists so this reaches stateful problems'
        per-op results too."""
        if isinstance(got, list) and isinstance(expected, list):
            return len(got) == len(expected) and all(
                CodeBenchmark._values_close(g, e) for g, e in zip(got, expected)
            )
        if (isinstance(got, (int, float)) and not isinstance(got, bool)
                and isinstance(expected, (int, float)) and not isinstance(expected, bool)
                and (isinstance(got, float) or isinstance(expected, float))):
            return math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)
        return got == expected

    @staticmethod
    def _run_harness(harness: str, payload: str, tests: list[dict],
                      timeout: int) -> list[dict]:
        """Run one streamed-result harness under one total timeout."""
        prefix = f"__LOCAL_AI_BENCH_RESULT_{secrets.token_hex(16)}__"
        harness_head, _, harness_tail = harness.rpartition("__RESULT_PREFIX__")
        harness = harness_head + repr(prefix) + harness_tail
        proc = subprocess.Popen(
            [sys.executable, "-c", harness],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()

        if timed_out:
            missing_error = "timeout"
        elif proc.returncode != 0:
            stderr_lines = (stderr or "").strip().splitlines()
            missing_error = stderr_lines[-1] if stderr_lines else "process failed"
        else:
            missing_error = "malformed output"
        return CodeBenchmark._parse_harness_output(stdout, prefix, tests, missing_error)

    @staticmethod
    def _parse_harness_output(stdout: str, prefix: str, tests: list[dict],
                              missing_error: str) -> list[dict]:
        """Parse framed per-test records and reject invalid protocol data."""
        raw_results = {}
        for line in (stdout or "").splitlines():
            if not line.startswith(prefix):
                continue
            try:
                raw = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                return CodeBenchmark._failed_test_results(tests, "malformed output")
            index = raw.get("index") if isinstance(raw, dict) else None
            if (isinstance(index, bool) or not isinstance(index, int)
                    or not 0 <= index < len(tests) or index in raw_results
                    or not isinstance(raw.get("ok"), bool)
                    or (raw["ok"] and "got" not in raw)
                    or (not raw["ok"] and "error" not in raw)):
                return CodeBenchmark._failed_test_results(tests, "malformed output")
            raw_results[index] = raw

        results = []
        for index, test in enumerate(tests):
            raw = raw_results.get(index)
            if raw is None:
                results.append({"passed": False, "got": None, "error": missing_error})
                continue
            if not raw.get("ok"):
                results.append({"passed": False, "got": None, "error": raw.get("error")})
            else:
                got = raw.get("got")
                results.append({"passed": CodeBenchmark._values_close(got, test["expected"]),
                                 "got": got, "error": None})
        return results

    @staticmethod
    def _failed_test_results(tests: list[dict], error: str) -> list[dict]:
        return [{"passed": False, "got": None, "error": error} for _ in tests]

    @staticmethod
    def execute_tests(code: str, function_name: str, tests: list[dict],
                       timeout: int = CODE_EXEC_TIMEOUT) -> list[dict]:
        """Run every test case in `tests` (each {"args": [...], "expected": ...})
        against `function_name` as defined by `code`. See `_run_harness` for
        the isolation/failure-handling contract.
        """
        harness = (
            "import json, sys\n\n"
            + code + "\n\n"
            "_args_list = json.loads(sys.stdin.read())\n"
            "for _index, _args in enumerate(_args_list):\n"
            "    try:\n"
            "        _result = {'index': _index, 'ok': True, 'got': " + function_name + "(*_args)}\n"
            "        _encoded = json.dumps(_result)\n"
            "    except Exception as _e:\n"
            "        _encoded = json.dumps({'index': _index, 'ok': False, 'error': str(_e)})\n"
            "    print('\\n' + __RESULT_PREFIX__ + _encoded, flush=True)\n"
        )
        payload = json.dumps([t["args"] for t in tests])
        return CodeBenchmark._run_harness(harness, payload, tests, timeout)

    @staticmethod
    def execute_stateful_tests(code: str, class_name: str, tests: list[dict],
                                timeout: int = CODE_EXEC_TIMEOUT) -> list[dict]:
        """Run every test case in `tests` (each {"init": [...] (optional,
        default []), "ops": [[method, args], ...], "expected": [...]})
        against `class_name` as defined by `code`: construct a fresh instance
        per test as class_name(*init), call each method in `ops` in order,
        and collect every return value into a list compared as a whole
        against `expected`. See `_run_harness` for the isolation/
        failure-handling contract.
        """
        harness = (
            "import json, sys\n\n"
            + code + "\n\n"
            "_scenarios = json.loads(sys.stdin.read())\n"
            "for _index, _scenario in enumerate(_scenarios):\n"
            "    try:\n"
            "        _obj = " + class_name + "(*_scenario.get('init', []))\n"
            "        _outputs = [getattr(_obj, _m)(*_a) for _m, _a in _scenario['ops']]\n"
            "        _result = {'index': _index, 'ok': True, 'got': _outputs}\n"
            "        _encoded = json.dumps(_result)\n"
            "    except Exception as _e:\n"
            "        _encoded = json.dumps({'index': _index, 'ok': False, 'error': str(_e)})\n"
            "    print('\\n' + __RESULT_PREFIX__ + _encoded, flush=True)\n"
        )
        payload = json.dumps([{"init": t.get("init", []), "ops": t["ops"]} for t in tests])
        return CodeBenchmark._run_harness(harness, payload, tests, timeout)

    @staticmethod
    def evaluate_question(question: dict, code: str | None) -> dict:
        """Run every visible+hidden test for `question` against `code` and
        summarize: {"correct": bool, "tests_passed": int, "tests_total": int,
        "error": str|None}. Empty/None `code` short-circuits to every test
        failing, without a subprocess call. Dispatches to
        `execute_stateful_tests` for a class-based problem (`class_name`
        present), `execute_tests` otherwise.
        """
        tests = question["visible_tests"] + question["hidden_tests"]
        if not code:
            return {"correct": False, "tests_passed": 0, "tests_total": len(tests), "error": "no code found"}

        if "class_name" in question:
            results = CodeBenchmark.execute_stateful_tests(code, question["class_name"], tests)
        else:
            results = CodeBenchmark.execute_tests(code, question["function_name"], tests)
        passed = sum(1 for r in results if r["passed"])
        first_error = next((r["error"] for r in results if r["error"]), None)
        return {
            "correct":      passed == len(tests),
            "tests_passed": passed,
            "tests_total":  len(tests),
            "error":        first_error,
        }

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[dict, str, bool]:
        prompt = CodeBenchmark.build_prompt(question)
        _, _, _, _, response_text, budget_nudged = engine.chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_ctx=config.ACCURACY_CONTEXT,
            num_predict=CodeBenchmark.CODE_NUM_PREDICT,
            check_loop=True,
            token_budget=config.ACC_TOKEN_BUDGET,
        )
        code = CodeBenchmark.extract_code(response_text)
        return CodeBenchmark.evaluate_question(question, code), response_text, budget_nudged

    @staticmethod
    def score(questions: list[dict], answers: dict) -> dict:
        """Tally correct/total overall and per category from a {question_id:
        evaluate_question_result_or_None} map. Pure, so it's directly
        testable."""
        by_category: dict[str, dict] = {}
        incorrect = []
        all_results = []
        correct = 0
        answered = 0

        for q in questions:
            qid, category = q["id"], q["category"]
            result = answers.get(qid)
            cat = by_category.setdefault(category, {"correct": 0, "total": 0})
            cat["total"] += 1
            if result is not None:
                answered += 1
            is_correct = result is not None and result["correct"]
            total_tests = len(q["visible_tests"]) + len(q["hidden_tests"])
            entry = {
                "id":           qid,
                "category":     category,
                "tests_passed": result["tests_passed"] if result else 0,
                "tests_total":  result["tests_total"] if result else total_tests,
                "error":        result["error"] if result else "unanswered",
            }
            if Shared.tally_accuracy_entry(entry, is_correct, cat, all_results, incorrect):
                correct += 1

        for cat in by_category.values():
            cat["accuracy_pct"] = round(100 * cat["correct"] / cat["total"], 1) if cat["total"] else 0.0

        total = len(questions)
        return {
            "correct":      correct,
            "total":        total,
            "answered":     answered,
            "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
            "by_category":  by_category,
            "incorrect":    incorrect,
            "all":          all_results,
        }

    def run(self, engine, models, questions=None, warmup_runs=config.WARMUP_RUNS, save_fn=None,
            answers_path: Path | None = None):  # pragma: no cover — orchestrates real engine runs
        questions = questions if questions is not None else CodeBenchmark.load_questions()

        def _rescore_partial(q, text):
            # Score whatever code streamed before the timeout rather than treating it as unanswered.
            return CodeBenchmark.evaluate_question(q, CodeBenchmark.extract_code(text))

        return Shared.run_accuracy_benchmark(
            section_label="Code", skip_label="code", question_noun="coding problems",
            data_path=CodeBenchmark.CODE_DATA_PATH, crash_cache_path=CodeBenchmark.CODE_CRASH_CACHE,
            models=models, questions=questions, warmup_runs=warmup_runs, engine=engine,
            ask_fn=lambda tag, q: CodeBenchmark._ask(engine, tag, q),
            rescore_partial_fn=_rescore_partial,
            score_fn=CodeBenchmark.score,
            save_fn=save_fn, answers_path=answers_path,
        )
