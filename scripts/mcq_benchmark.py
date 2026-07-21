"""mcq_benchmark.py — multiple-choice accuracy benchmark: each model answers
every question in scripts/data/mcq_questions.json once at temperature 0,
scored right/wrong against the dataset's known answer and broken down by
category.
"""

import json
import re
from pathlib import Path

import config
from shared import Shared


class MCQBenchmark:
    MCQ_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "mcq_questions.json"

    # Records models that crashed the engine's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    MCQ_CRASH_CACHE = Path(".mcq_crash_cache.json")

    # Unbounded (-1): a fixed token cap risks truncating a reasoning model's
    # answer. The wall-clock timeout in the engine's chat is the real bound.
    MCQ_NUM_PREDICT = -1

    # Uppercase-only: models answer in uppercase ("B"), so scanning for A-D as
    # written avoids false hits on lowercase words/contractions a case-
    # insensitive scan would catch (the "d" in "I'd", the article "a").
    _LETTER_RE = re.compile(r"\b([A-D])\b")

    @staticmethod
    def load_questions(path: Path = MCQ_DATA_PATH) -> list[dict]:
        return json.loads(Path(path).read_text())

    @staticmethod
    def build_prompt(question: dict) -> str:
        choices_text = "\n".join(f"{letter}. {text}" for letter, text in question["choices"].items())
        return (
            f"{question['prompt']}\n\n{choices_text}\n\n"
            "Respond with only the letter of the correct answer."
        )

    @staticmethod
    def parse_answer(response_text: str, valid_choices) -> str | None:
        """Extract the model's chosen letter from free-form text, or None.
        Takes the *last* standalone valid letter, not the first — a reasoning
        model walks through rejected options by letter before its final
        choice. Same reasoning as MathBenchmark.parse_answer."""
        if not response_text:
            return None
        valid = {c.upper() for c in valid_choices}

        # Handle a bare single letter case-insensitively (before the
        # uppercase-only scan below), so a lowercase "b" or "(b)" still counts.
        stripped = response_text.strip().strip(".()[]:*").strip()
        if len(stripped) == 1 and stripped.upper() in valid:
            return stripped.upper()

        found = None
        for match in MCQBenchmark._LETTER_RE.finditer(response_text):
            letter = match.group(1)
            if letter in valid:
                found = letter
        return found

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[str | None, str]:
        prompt = MCQBenchmark.build_prompt(question)
        _, _, _, _, response_text = engine.chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_ctx=config.ACCURACY_CONTEXT,
            num_predict=MCQBenchmark.MCQ_NUM_PREDICT,
            check_loop=True,
        )
        return MCQBenchmark.parse_answer(response_text, question["choices"].keys()), response_text

    @staticmethod
    def score(questions: list[dict], answers: dict) -> dict:
        """Tally correct/total overall and per category from a {question_id:
        given_letter_or_None} map. Pure, so it's directly testable."""
        by_category: dict[str, dict] = {}
        incorrect = []
        correct = 0
        answered = 0

        for q in questions:
            qid, category, expected = q["id"], q["category"], q["answer"]
            given = answers.get(qid)
            cat = by_category.setdefault(category, {"correct": 0, "total": 0})
            cat["total"] += 1
            if given is not None:
                answered += 1
            is_correct = given == expected
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
        questions = questions if questions is not None else MCQBenchmark.load_questions()
        return Shared.run_accuracy_benchmark(
            section_label="MCQ", skip_label="MCQ", question_noun="MCQ questions",
            data_path=MCQBenchmark.MCQ_DATA_PATH, crash_cache_path=MCQBenchmark.MCQ_CRASH_CACHE,
            models=models, questions=questions, warmup_runs=warmup_runs, engine=engine,
            ask_fn=lambda tag, q: MCQBenchmark._ask(engine, tag, q),
            rescore_partial_fn=lambda q, text: MCQBenchmark.parse_answer(text, q["choices"].keys()),
            score_fn=MCQBenchmark.score,
            save_fn=save_fn, answers_path=answers_path,
        )
