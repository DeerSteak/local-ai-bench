"""Tool-calling accuracy benchmark — see docs/workloads.md#tool-use."""

import json
from pathlib import Path

from scripts.runtime import config
from scripts.runtime.shared import Shared
from scripts.workloads.accuracy_scoring import score_question_bank, validate_question_bank


class ToolBenchmark:
    TOOL_DATA_PATH = Path(__file__).with_name("data") / "tool_questions.json"

    TOOL_CRASH_CACHE = Path(".tool_crash_cache.json")  # see docs/project-structure.md

    # -1 delegates the finite per-pass limits to chat_tools' token_budget split.
    TOOL_NUM_PREDICT = -1

    @staticmethod
    def load_questions(path: Path = TOOL_DATA_PATH) -> list[dict]:
        return validate_question_bank(
            json.loads(Path(path).read_text(encoding="utf-8")),
            ("prompt", "tools", "expected"),
        )

    @staticmethod
    def _coerce(value):
        """Coerce a numeric string to a number so "20" matches an expected 20."""
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
    def _normalize_string(value: str) -> str:
        return value.strip().casefold().rstrip(".!?")

    @staticmethod
    def _values_equal(given, expected, strict: bool, unordered_keys: frozenset = frozenset(),
                      normalized_string_keys: frozenset = frozenset(),
                      normalize_string: bool = False) -> bool:
        """Recursively compare `given` against `expected` — see docs/workloads.md#tool-use
        for `unordered_keys`/`normalized_string_keys` semantics."""
        if isinstance(expected, dict):
            if not isinstance(given, dict):
                return False
            if strict and given.keys() != expected.keys():
                return False
            for key, exp in expected.items():
                if key not in given:
                    return False
                if key in unordered_keys and isinstance(exp, list):
                    if not ToolBenchmark._multiset_equal(
                            given[key], exp, strict, unordered_keys,
                            normalized_string_keys, key in normalized_string_keys):
                        return False
                elif not ToolBenchmark._values_equal(
                        given[key], exp, strict, unordered_keys,
                        normalized_string_keys, key in normalized_string_keys):
                    return False
            return True
        if isinstance(expected, list):
            if not isinstance(given, list) or len(given) != len(expected):
                return False
            return all(ToolBenchmark._values_equal(
                g, e, strict, unordered_keys, normalized_string_keys, normalize_string,
            ) for g, e in zip(given, expected))
        # bool is a subclass of int in Python (True == 1), so without this
        # check a boolean argument would wrongly match a numeric one.
        if isinstance(given, bool) != isinstance(expected, bool):
            return False
        if normalize_string and isinstance(given, str) and isinstance(expected, str):
            return ToolBenchmark._normalize_string(given) == ToolBenchmark._normalize_string(expected)
        return ToolBenchmark._coerce(given) == ToolBenchmark._coerce(expected)

    @staticmethod
    def _multiset_equal(given, expected: list, strict: bool, unordered_keys: frozenset,
                        normalized_string_keys: frozenset = frozenset(),
                        normalize_string: bool = False) -> bool:
        """Order-insensitive list match: every expected element is matched to
        a distinct given element (recursively), regardless of position."""
        if not isinstance(given, list) or len(given) != len(expected):
            return False

        def match(expected_i: int, used: frozenset[int]) -> bool:
            if expected_i == len(expected):
                return True
            return any(
                given_i not in used
                and ToolBenchmark._values_equal(
                    value, expected[expected_i], strict, unordered_keys,
                    normalized_string_keys, normalize_string,
                )
                and match(expected_i + 1, used | {given_i})
                for given_i, value in enumerate(given)
            )

        return match(0, frozenset())

    @staticmethod
    def _args_match(given: dict, expected: dict, allow_extra: bool = True,
                    unordered_keys=(), normalized_string_keys=()) -> bool:
        """Loose-equality argument match — see docs/workloads.md#tool-use for
        subset-vs-strict matching."""
        if not isinstance(given, dict):
            return False
        return ToolBenchmark._values_equal(
            given, expected, strict=not allow_extra,
            unordered_keys=frozenset(unordered_keys),
            normalized_string_keys=frozenset(normalized_string_keys),
        )

    @staticmethod
    def evaluate_question(question: dict, tool_calls: list | None) -> dict:
        """Score `tool_calls` against `question["expected"]` — see docs/workloads.md#tool-use
        for decline-vs-call semantics."""
        expected = question["expected"]
        calls = tool_calls or []

        if not expected.get("call"):
            return {"correct": len(calls) == 0}

        if len(calls) != 1:
            return {"correct": False}
        first = calls[0]
        correct = (not first.get("incomplete")
                   and first.get("name") == expected["name"]
                   and ToolBenchmark._args_match(
                       first.get("arguments", {}), expected["arguments"],
                       allow_extra=not expected.get("strict_arguments", False),
                       unordered_keys=expected.get("unordered_keys", ()),
                       normalized_string_keys=expected.get("normalized_string_keys", ()),
                   ))
        return {"correct": correct}

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[dict, str, bool]:
        measurement = engine.chat_tools(
            tag, [{"role": "user", "content": question["prompt"]}],
            tools=question["tools"], timeout=config.ACC_TIMEOUT,
            num_ctx=config.ACCURACY_CONTEXT,
            num_predict=ToolBenchmark.TOOL_NUM_PREDICT, check_loop=True,
            token_budget=config.ACC_TOKEN_BUDGET,
        )
        # Keep prose alongside tool calls — a decline is prose with no calls at all.
        if measurement.tool_calls and measurement.response_text:
            raw = json.dumps({"tool_calls": measurement.tool_calls,
                              "text": measurement.response_text})
        elif measurement.tool_calls:
            raw = json.dumps(measurement.tool_calls)
        else:
            raw = measurement.response_text
        return (ToolBenchmark.evaluate_question(question, measurement.tool_calls),
                raw, measurement.budget_nudged)

    @staticmethod
    def rescore_partial_fn(question: dict, partial_text: str) -> dict:
        """Best-effort rescore of a timed-out question, falling back to [] (a decline) if it won't parse."""
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
        evaluate_question_result_or_None} map."""
        return score_question_bank(
            questions, answers,
            lambda question, result: (
                result is not None, result is not None and result["correct"],
                {"id": question["id"], "category": question["category"]},
            ),
        )

    def run(self, engine, models, questions=None, warmup_runs=config.WARMUP_RUNS, save_fn=None,
            answers_path: Path | None = None,
            telemetry=None, journal=None):  # pragma: no cover — orchestrates real engine runs
        questions = questions if questions is not None else ToolBenchmark.load_questions()
        return Shared.run_accuracy_benchmark(
            section_label="Tool", skip_label="tool", question_noun="tool question",
            data_path=ToolBenchmark.TOOL_DATA_PATH, crash_cache_path=ToolBenchmark.TOOL_CRASH_CACHE,
            models=models, questions=questions, warmup_runs=warmup_runs, engine=engine,
            ask_fn=lambda tag, q: ToolBenchmark._ask(engine, tag, q),
            rescore_partial_fn=ToolBenchmark.rescore_partial_fn,
            score_fn=ToolBenchmark.score,
            save_fn=save_fn, answers_path=answers_path, progress_stage="tool",
            requires_tool_calls=True, telemetry=telemetry, journal=journal,
        )
