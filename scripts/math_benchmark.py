"""math_benchmark.py — math word-problem accuracy benchmark: each model
answers every question in scripts/data/math_questions.json once at
temperature 0, scored right/wrong against the dataset's known numeric answer
(within its per-question tolerance) and broken down by category.
"""

import json
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
    _CONCLUSION_RE = re.compile(
        rf"\b(?:therefore|thus|so)\b[\s\S]{{0,500}}?"
        rf"\b(?:is|equals|has|gives|yields)\b\s*(?:approximately\s+)?({_NUMBER_PATTERN})",
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

        conclusions = list(MathBenchmark._CONCLUSION_RE.finditer(response_text))
        if conclusions:
            return MathBenchmark._to_float(conclusions[-1].group(1))

        matches = MathBenchmark._NUMBER_RE.findall(response_text)
        return MathBenchmark._to_float(matches[-1]) if matches else None

    @staticmethod
    def _to_float(value: str) -> float | None:
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None

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
