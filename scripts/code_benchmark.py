"""code_benchmark.py — coding accuracy benchmark: each model answers every
problem in scripts/data/code_problems.json once at temperature 0, then that
answer is run against the problem's visible and hidden test cases in an
isolated subprocess. A problem counts as correct only if every one of its
test cases passes. Scored overall and broken down by category, same shape
as MCQBenchmark/MathBenchmark.

Two problem shapes:
- Function problems (`function_name`): the model writes one function; each
  test is {"args": [...], "expected": ...}, called as function_name(*args).
- Stateful problems (`class_name`): the model writes a class; each test is
  {"init": [...] (optional, default []), "ops": [[method, args], ...],
  "expected": [...]} — a fresh instance per test, constructed as
  class_name(*init), then each method called in sequence and every return
  value collected and compared to `expected` as a whole.

`visible_tests` are rendered into the prompt as worked examples (the model
is meant to see them, same as example I/O in a real problem statement);
`hidden_tests` are never shown and only used for scoring. Both are run
together at grading time, so a model can't game the split by memorizing
just the visible cases.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import config
from shared import Shared


class CodeBenchmark:
    CODE_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "code_problems.json"

    # Records models that crashed Ollama's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    CODE_CRASH_CACHE = Path(".code_crash_cache.json")

    # Unbounded (-1): a fixed token cap risks truncating a reasoning model's
    # answer. The wall-clock timeout in Shared.ollama_chat is the real bound.
    CODE_NUM_PREDICT = -1

    # Wall-clock budget for running a model's generated code against one
    # problem's tests in the execute_tests() subprocess. Not a security sandbox
    # — just enough isolation that bad code (infinite loop, crash, stray print)
    # can't hang or corrupt the benchmark process. Generous for these problems'
    # scale, so a real timeout is treated as the code being wrong.
    CODE_EXEC_TIMEOUT = 5

    # Pulls the code out of a fenced block (```python ... ``` or ``` ... ```);
    # DOTALL so the body can span multiple lines.
    _FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

    @staticmethod
    def load_questions(path: Path = CODE_DATA_PATH) -> list[dict]:
        return json.loads(Path(path).read_text())

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
        """Pull the model's code out of its free-form reply.

        Prefers a fenced code block if present (the requested format); falls
        back to the whole stripped reply for models that ignore the fencing
        instruction and write bare code.
        """
        if not response_text:
            return ""
        match = CodeBenchmark._FENCE_RE.search(response_text)
        if match:
            return match.group(1).strip()
        return response_text.strip()

    @staticmethod
    def _run_harness(harness: str, payload: str, tests: list[dict],
                      timeout: int) -> list[dict]:
        """Run `harness` (candidate code plus a driver that prints one JSON
        {"ok": bool, "got"|"error": ...} object per test) against `payload`
        on stdin, in a separate subprocess so generated code can't hang,
        crash, or leak into the benchmark process. Returns one {"passed":
        bool, "got": ..., "error": str|None} entry per test, in order. A
        subprocess-level failure (syntax error, timeout, non-serializable
        return, ...) marks every test failed with the same error rather than
        raising.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-c", harness],
                input=payload, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return [{"passed": False, "got": None, "error": "timeout"} for _ in tests]

        if proc.returncode != 0 or not proc.stdout.strip():
            stderr_lines = (proc.stderr or "").strip().splitlines()
            err = stderr_lines[-1] if stderr_lines else "process failed"
            return [{"passed": False, "got": None, "error": err} for _ in tests]

        try:
            raw_results = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return [{"passed": False, "got": None, "error": "malformed output"} for _ in tests]

        results = []
        for test, raw in zip(tests, raw_results):
            if not raw.get("ok"):
                results.append({"passed": False, "got": None, "error": raw.get("error")})
            else:
                got = raw.get("got")
                results.append({"passed": got == test["expected"], "got": got, "error": None})
        return results

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
            "_results = []\n"
            "for _args in _args_list:\n"
            "    try:\n"
            "        _results.append({'ok': True, 'got': " + function_name + "(*_args)})\n"
            "    except Exception as _e:\n"
            "        _results.append({'ok': False, 'error': str(_e)})\n"
            "print(json.dumps(_results))\n"
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
            "_results = []\n"
            "for _scenario in _scenarios:\n"
            "    try:\n"
            "        _obj = " + class_name + "(*_scenario.get('init', []))\n"
            "        _outputs = [getattr(_obj, _m)(*_a) for _m, _a in _scenario['ops']]\n"
            "        _results.append({'ok': True, 'got': _outputs})\n"
            "    except Exception as _e:\n"
            "        _results.append({'ok': False, 'error': str(_e)})\n"
            "print(json.dumps(_results))\n"
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
    def _ask(tag: str, question: dict) -> tuple[dict, str]:
        prompt = CodeBenchmark.build_prompt(question)
        _, _, _, _, response_text = Shared.ollama_chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.RUN_TIMEOUT, num_predict=CodeBenchmark.CODE_NUM_PREDICT,
        )
        code = CodeBenchmark.extract_code(response_text)
        return CodeBenchmark.evaluate_question(question, code), response_text

    @staticmethod
    def score(questions: list[dict], answers: dict) -> dict:
        """Tally correct/total overall and per category from a {question_id:
        evaluate_question_result_or_None} map. Pure, so it's directly
        testable."""
        by_category: dict[str, dict] = {}
        incorrect = []
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
            if is_correct:
                correct += 1
                cat["correct"] += 1
            else:
                total_tests = len(q["visible_tests"]) + len(q["hidden_tests"])
                incorrect.append({
                    "id":           qid,
                    "category":     category,
                    "tests_passed": result["tests_passed"] if result else 0,
                    "tests_total":  result["tests_total"] if result else total_tests,
                    "error":        result["error"] if result else "unanswered",
                })

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
        }

    def run(self, models, questions=None, warmup_runs=config.WARMUP_RUNS, save_fn=None,
            answers_path: Path | None = None):  # pragma: no cover — orchestrates real Ollama runs
        results = {}
        answers_out: dict = {}
        questions = questions if questions is not None else CodeBenchmark.load_questions()

        if not Shared.ollama_available():
            Shared.err("Ollama server not reachable — skipping code benchmark")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(CodeBenchmark.CODE_CRASH_CACHE)
        bank_hash = Shared.file_hash(CodeBenchmark.CODE_DATA_PATH)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"Code: {label}")

            if not Shared.ollama_reachable_or_abort():
                break

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, CodeBenchmark.CODE_CRASH_CACHE,
                                                       expected_bank_hash=bank_hash)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                if not Shared.warmup_model(tag, label, config.CONTEXT_LENGTHS[0], warmup_runs,
                                           crash_cache, CodeBenchmark.CODE_CRASH_CACHE,
                                           crash_extra={"bank_hash": bank_hash}):
                    Shared.unload_model(tag)
                    continue

                Shared.log(f"Answering {len(questions)} coding problems ...")
                answers: dict[str, dict | None] = {}
                raw_responses: dict[str, str] = {}
                stopped_early = None

                for i, q in enumerate(questions):
                    samples, status = Shared.run_measured_calls(
                        1, lambda run_i, q=q: CodeBenchmark._ask(tag, q), tag, crash_cache,
                        CodeBenchmark.CODE_CRASH_CACHE, f"answering {q['id']}",
                        crash_extra={"bank_hash": bank_hash})
                    result, raw = samples[0] if samples else (None, "")
                    answers[q["id"]] = result
                    raw_responses[q["id"]] = raw

                    if status == "timed_out":
                        Shared.err(f"Skipping remaining questions for {label}")
                        stopped_early = "timed_out"
                        break
                    if status == "crashed":
                        stopped_early = "crashed"
                        break

                    if (i + 1) % 10 == 0:
                        Shared.log(f"  {i+1}/{len(questions)} answered ...")

                scored = CodeBenchmark.score(questions, answers)
                answers_out[short] = {
                    "label": label,
                    "incorrect": [
                        {**entry, "raw_response": raw_responses.get(entry["id"], "")}
                        for entry in scored["incorrect"]
                    ],
                }
                results[short] = {"label": label, **scored}

                if stopped_early == "timed_out":
                    results[short]["timed_out"] = True
                elif stopped_early == "crashed":
                    crashed_at = crash_cache.get(tag, {}).get("crashed_at", "an earlier run")
                    results[short]["crashed"] = True
                    results[short]["crashed_at"] = crashed_at

                Shared.ok(f"{label}: {scored['accuracy_pct']:.1f}% "
                          f"({scored['correct']}/{scored['total']})")

                Shared.log(f"Unloading {label} ...")
                Shared.unload_model(tag)
                Shared.wait_until_unloaded(tag)
            finally:
                if save_fn:
                    save_fn(results)
                if answers_path:
                    Shared.write_answers_sidecar(answers_path, answers_out)

        return results
