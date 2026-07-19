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

    # Matches an optionally-negative, optionally-decimal number, with commas
    # allowed as thousands separators (stripped before parsing).
    _NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

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
        """Extract the model's numeric answer from free-form text, or None.

        Takes the *last* number, not the first — a model reasoning out loud
        ("347 + 589 = 936, so the answer is 936") states its final answer
        last, and intermediate numbers shouldn't be mistaken for it.
        """
        if not response_text:
            return None
        matches = MathBenchmark._NUMBER_RE.findall(response_text)
        if not matches:
            return None
        last = matches[-1].replace(",", "")
        if not last or last == "-":
            return None
        try:
            return float(last)
        except ValueError:
            return None

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[float | None, str]:
        prompt = MathBenchmark.build_prompt(question)
        _, _, _, _, response_text = engine.chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_predict=MathBenchmark.MATH_NUM_PREDICT,
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
            if is_correct:
                correct += 1
                cat["correct"] += 1
            else:
                incorrect.append({"id": qid, "category": category, "given": given, "expected": expected})

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
