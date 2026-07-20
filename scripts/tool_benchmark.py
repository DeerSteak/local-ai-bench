"""tool_benchmark.py — tool-calling accuracy benchmark: each model is offered
an OpenAI-style tools array per question in scripts/data/tool_questions.json
and asked to answer once at temperature 0, scored on whether it called the
right tool with the right arguments (or correctly declined to call anything
when none of the offered tools fit), broken down by category. Same result
shape as MCQBenchmark/MathBenchmark/CodeBenchmark.
"""

import json
from pathlib import Path

import config
from shared import Shared


class ToolBenchmark:
    TOOL_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "tool_questions.json"

    # Records models that crashed the engine's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    TOOL_CRASH_CACHE = Path(".tool_crash_cache.json")

    # Unbounded (-1): a fixed token cap risks truncating a reasoning model's
    # answer. The wall-clock timeout in the engine's chat is the real bound.
    TOOL_NUM_PREDICT = -1

    @staticmethod
    def load_questions(path: Path = TOOL_DATA_PATH) -> list[dict]:
        return json.loads(Path(path).read_text())

    @staticmethod
    def _coerce(value):
        """Coerce a numeric string to a number so a model emitting "20" for a
        numeric argument still matches an expected 20. Non-numeric strings and
        everything else pass through unchanged."""
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return value

    @staticmethod
    def _values_equal(given, expected, strict: bool, unordered_keys: frozenset = frozenset()) -> bool:
        """Recursively compare `given` against `expected`, coercing numeric
        strings to numbers at every level (not just the top one) so a nested
        object or array-of-objects argument gets the same tolerant matching
        as a flat one. `unordered_keys` names dict keys (matched by name at
        any depth) whose list value should be compared as a multiset rather
        than positionally, for genuinely order-insensitive arguments like a
        set of labels or recipients."""
        if isinstance(expected, dict):
            if not isinstance(given, dict):
                return False
            if strict and given.keys() != expected.keys():
                return False
            for key, exp in expected.items():
                if key not in given:
                    return False
                if key in unordered_keys and isinstance(exp, list):
                    if not ToolBenchmark._multiset_equal(given[key], exp, strict, unordered_keys):
                        return False
                elif not ToolBenchmark._values_equal(given[key], exp, strict, unordered_keys):
                    return False
            return True
        if isinstance(expected, list):
            if not isinstance(given, list) or len(given) != len(expected):
                return False
            return all(ToolBenchmark._values_equal(g, e, strict, unordered_keys) for g, e in zip(given, expected))
        # bool is a subclass of int in Python (True == 1), so without this
        # check a boolean argument would wrongly match a numeric one.
        if isinstance(given, bool) != isinstance(expected, bool):
            return False
        return ToolBenchmark._coerce(given) == ToolBenchmark._coerce(expected)

    @staticmethod
    def _multiset_equal(given, expected: list, strict: bool, unordered_keys: frozenset) -> bool:
        """Order-insensitive list match: every expected element is matched to
        a distinct given element (recursively), regardless of position."""
        if not isinstance(given, list) or len(given) != len(expected):
            return False
        remaining = list(given)
        for exp in expected:
            for i, g in enumerate(remaining):
                if ToolBenchmark._values_equal(g, exp, strict, unordered_keys):
                    del remaining[i]
                    break
            else:
                return False
        return True

    @staticmethod
    def _args_match(given: dict, expected: dict, allow_extra: bool = True, unordered_keys=()) -> bool:
        """Loose-equality argument match: every expected key present with an
        equal value, numeric strings coerced to numbers first so "20" == 20.
        Extra keys are allowed unless the question requests strict matching."""
        if not isinstance(given, dict):
            return False
        return ToolBenchmark._values_equal(given, expected, strict=not allow_extra,
                                            unordered_keys=frozenset(unordered_keys))

    @staticmethod
    def evaluate_question(question: dict, tool_calls: list | None) -> dict:
        """Score `tool_calls` against `question["expected"]`: {"correct": bool}.
        A decline case (`call` is False) is correct only if nothing was called;
        a call case is correct only if exactly one call names the expected tool
        with matching arguments. None/empty tool_calls means the model declined."""
        expected = question["expected"]
        calls = tool_calls or []

        if not expected.get("call"):
            return {"correct": len(calls) == 0}

        if len(calls) != 1:
            return {"correct": False}
        first = calls[0]
        correct = (first.get("name") == expected["name"]
                   and ToolBenchmark._args_match(
                       first.get("arguments") or {}, expected["arguments"],
                       allow_extra=not expected.get("strict_arguments", False),
                       unordered_keys=expected.get("unordered_keys", ()),
                   ))
        return {"correct": correct}

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[dict, str]:
        _, _, _, _, response_text, tool_calls = engine.chat_tools(
            tag, [{"role": "user", "content": question["prompt"]}],
            tools=question["tools"], timeout=config.ACC_TIMEOUT,
            num_predict=ToolBenchmark.TOOL_NUM_PREDICT, check_loop=True,
        )
        return ToolBenchmark.evaluate_question(question, tool_calls), json.dumps(tool_calls)

    @staticmethod
    def rescore_partial_fn(question: dict, partial_text: str) -> dict:
        """Best-effort rescore of a timed-out question: try to parse a
        tool-call list out of whatever text streamed before the cutoff,
        falling back to [] (a decline) if it won't parse."""
        try:
            parsed = json.loads(partial_text)
        except (json.JSONDecodeError, TypeError):
            parsed = []
        if not isinstance(parsed, list):
            parsed = []
        return ToolBenchmark.evaluate_question(question, parsed)

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
                incorrect.append({"id": qid, "category": category})

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

    def run(self, engine, models, questions=None, warmup_runs=config.WARMUP_RUNS, save_fn=None,
            answers_path: Path | None = None):  # pragma: no cover — orchestrates real engine runs
        questions = questions if questions is not None else ToolBenchmark.load_questions()
        return Shared.run_accuracy_benchmark(
            section_label="Tool", skip_label="tool", question_noun="tool question",
            data_path=ToolBenchmark.TOOL_DATA_PATH, crash_cache_path=ToolBenchmark.TOOL_CRASH_CACHE,
            models=models, questions=questions, warmup_runs=warmup_runs, engine=engine,
            ask_fn=lambda tag, q: ToolBenchmark._ask(engine, tag, q),
            rescore_partial_fn=ToolBenchmark.rescore_partial_fn,
            score_fn=ToolBenchmark.score,
            save_fn=save_fn, answers_path=answers_path,
        )
