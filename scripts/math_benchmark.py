"""math_benchmark.py — math word-problem accuracy benchmark: each model
answers every question in scripts/data/math_questions.json once at
temperature 0, scored right/wrong against the dataset's known numeric answer
(within its per-question tolerance) and broken down by category.
"""

import json
import math
import re
from pathlib import Path

import config
from shared import Shared


class MathBenchmark:
    MATH_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "math_questions.json"

    # Records models that crashed the engine's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    MATH_CRASH_CACHE = Path(".math_crash_cache.json")

    # Unbounded (-1): a fixed token cap risks truncating a reasoning model's
    # answer. The wall-clock timeout in the engine's chat is the real bound.
    MATH_NUM_PREDICT = -1

    _NUMBER_PATTERN = r"-?\d[\d,]*\.?\d*"
    _NUMBER_RE = re.compile(_NUMBER_PATTERN)
    _BARE_NUMBER_RE = re.compile(rf"^\s*({_NUMBER_PATTERN})\s*%?\s*$")
    _BOXED_RE = re.compile(rf"\\boxed\s*\{{\s*({_NUMBER_PATTERN})\s*%?\s*\}}", re.IGNORECASE)
    _ANSWER_RE = re.compile(
        rf"\b(?:final\s+answer|answer)\b\s*(?:is|:|=)\s*"
        rf"(?:actually\s+)?(?:\$|\\?\(|\\?\[)?\s*({_NUMBER_PATTERN})",
        re.IGNORECASE,
    )
    _LEADING_NUMBER_RE = re.compile(
        rf"^\s*({_NUMBER_PATTERN})[ \t]*%?[ \t]*(?:\r?\n|$)",
    )
    _CONCLUSION_RE = re.compile(
        rf"(?:^|(?<=[.!?])|(?<=\n))[ \t]*(?:therefore|thus|so)\b[\s\S]{{0,500}}?"
        rf"\b(?:is|equals|has|gives|yields)\b\s*(?:approximately\s+)?"
        rf"({_NUMBER_PATTERN})(?![\d,.])",
        re.IGNORECASE,
    )
    _EQUAL_RESULT_RE = re.compile(
        rf"=\s*(?:approximately\s+)?({_NUMBER_PATTERN})(?![\d,.])",
        re.IGNORECASE,
    )

    @staticmethod
    def load_questions(path: Path = MATH_DATA_PATH) -> list[dict]:
        return json.loads(Path(path).read_text())

    @staticmethod
    def build_prompt(question: dict) -> str:
        return (
            f"{question['prompt']}\n\n"
            "Respond with only the final numeric answer, with no other text."
        )

    @staticmethod
    def parse_answer(response_text: str) -> float | None:
        """Extract a stated numeric result from free-form text, or None."""
        if not response_text:
            return None

        bare = MathBenchmark._BARE_NUMBER_RE.fullmatch(response_text)
        if bare:
            return MathBenchmark._to_float(bare.group(1))

        structured = [
            (match.start(), match.group(1))
            for pattern in (MathBenchmark._BOXED_RE, MathBenchmark._ANSWER_RE)
            for match in pattern.finditer(response_text)
        ]
        if structured:
            return MathBenchmark._to_float(max(structured, key=lambda candidate: candidate[0])[1])

        fallback_matches = MathBenchmark._NUMBER_RE.findall(response_text)
        fallback = fallback_matches[-1] if fallback_matches else None
        stated = []
        leading = MathBenchmark._LEADING_NUMBER_RE.match(response_text)
        if leading:
            leading_value = MathBenchmark._to_float(leading.group(1))
            fallback_value = MathBenchmark._to_float(fallback) if fallback is not None else None
            equality_value = MathBenchmark._last_completed_equality_value(response_text)
            if leading_value in (fallback_value, equality_value):
                stated.append((leading.start(1), leading.group(1)))
        for conclusion in MathBenchmark._CONCLUSION_RE.finditer(response_text):
            value = MathBenchmark._conclusion_value(response_text, conclusion)
            if value is not None:
                stated.append((conclusion.start(1), value))
        if stated:
            return MathBenchmark._to_float(max(stated, key=lambda candidate: candidate[0])[1])

        return MathBenchmark._to_float(fallback) if fallback is not None else None

    @staticmethod
    def _to_float(value: str) -> float | None:
        try:
            parsed = float(value.replace(",", ""))
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _conclusion_value(response_text: str, match: re.Match) -> str | None:
        line_start = response_text.rfind("\n", 0, match.start()) + 1
        line_prefix = response_text[line_start:match.start()]
        if re.fullmatch(r"\s*(?:[-*]\s*)?\d+\.", line_prefix):
            return None

        tail_start = match.end(1)
        cursor = tail_start
        if cursor < len(response_text) and response_text[cursor] == "%":
            cursor += 1
        while cursor < len(response_text) and response_text[cursor] in " \t":
            cursor += 1
        if cursor == len(response_text) or response_text[cursor] not in "+-*/^=":
            return match.group(1)

        clause_end = MathBenchmark._clause_end(response_text, cursor)
        clause = response_text[cursor:clause_end]
        results = list(MathBenchmark._EQUAL_RESULT_RE.finditer(clause))
        if not results:
            return None
        result = results[-1]
        result_end = result.end(1)
        while result_end < len(clause) and clause[result_end] in "% \t":
            result_end += 1
        if result_end < len(clause) and clause[result_end] in "+-*/^=":
            return None
        return result.group(1)

    @staticmethod
    def _clause_end(response_text: str, start: int) -> int:
        for index in range(start, len(response_text)):
            char = response_text[index]
            if char in "\n;?!":
                return index
            if char == ".":
                before_digit = index > 0 and response_text[index - 1].isdigit()
                after_digit = index + 1 < len(response_text) and response_text[index + 1].isdigit()
                if not (before_digit and after_digit):
                    return index
        return len(response_text)

    @staticmethod
    def _last_completed_equality_value(response_text: str) -> float | None:
        completed = []
        for result in MathBenchmark._EQUAL_RESULT_RE.finditer(response_text):
            cursor = result.end(1)
            while cursor < len(response_text) and response_text[cursor] in "% \t":
                cursor += 1
            if cursor < len(response_text) and response_text[cursor] in "+-*/^=":
                continue
            completed.append(result.group(1))
        return MathBenchmark._to_float(completed[-1]) if completed else None

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[float | None, str]:
        prompt = MathBenchmark.build_prompt(question)
        _, _, _, _, response_text = engine.chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_ctx=config.ACCURACY_CONTEXT,
            num_predict=MathBenchmark.MATH_NUM_PREDICT,
            check_loop=True,
        )
        return MathBenchmark.parse_answer(response_text), response_text

    @staticmethod
    def score(questions: list[dict], answers: dict) -> dict:
        """Tally correct/total overall and per category from a {question_id:
        given_number_or_None} map, comparing each answer against its own
        question's tolerance. Pure, so it's directly testable."""
        by_category: dict[str, dict] = {}
        incorrect = []
        all_results = []
        correct = 0
        answered = 0

        for q in questions:
            qid, category, expected = q["id"], q["category"], q["answer"]
            tolerance = q.get("tolerance", 0)
            given = answers.get(qid)
            cat = by_category.setdefault(category, {"correct": 0, "total": 0})
            cat["total"] += 1
            if given is not None:
                answered += 1
            is_correct = given is not None and abs(given - expected) <= tolerance
            entry = {"id": qid, "category": category, "given": given, "expected": expected}
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
        questions = questions if questions is not None else MathBenchmark.load_questions()
        return Shared.run_accuracy_benchmark(
            section_label="Math", skip_label="math", question_noun="math questions",
            data_path=MathBenchmark.MATH_DATA_PATH, crash_cache_path=MathBenchmark.MATH_CRASH_CACHE,
            models=models, questions=questions, warmup_runs=warmup_runs, engine=engine,
            ask_fn=lambda tag, q: MathBenchmark._ask(engine, tag, q),
            rescore_partial_fn=lambda q, text: MathBenchmark.parse_answer(text),
            score_fn=MathBenchmark.score,
            save_fn=save_fn, answers_path=answers_path,
        )
