"""Multiple-choice accuracy benchmark — see docs/workloads.md#accuracy."""

import json
import re
from pathlib import Path

import config
from shared import Shared


class MCQBenchmark:
    MCQ_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "mcq_questions.json"

    MCQ_CRASH_CACHE = Path(".mcq_crash_cache.json")  # see docs/project-structure.md

    # -1 delegates the finite per-pass limits to chat's token_budget split.
    MCQ_NUM_PREDICT = -1

    # Uppercase-only avoids ordinary words; leading article A is filtered separately.
    _LETTER_RE = re.compile(r"\b([A-D])\b")
    _BOXED_RE = re.compile(r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE)
    _BOXED_TEXT_RE = re.compile(
        r"\\boxed\s*\{\s*\\(?:text|mathrm|mathbf|operatorname)\s*"
        r"\{\s*([A-D])\s*\}\s*\}", re.IGNORECASE,
    )
    _TAGGED_RE = re.compile(
        r"<(answer|final_answer|final)>\s*([A-D])\s*</\1>", re.IGNORECASE,
    )
    _JSON_ANSWER_RE = re.compile(
        r'["\'](?:answer|final_answer)["\']\s*:\s*["\']([A-D])["\']', re.IGNORECASE,
    )
    _DECORATED_ANSWER_RE = re.compile(
        r"\b(?:final\s+answer|correct\s+(?:answer|choice)|the\s+answer|my\s+answer|answer)\b"
        r"\s*(?:is\b\s*:?|:|=)\s*(?:option|choice)?\s*[*_`]{1,3}\s*[\[(]?([A-D])\b",
        re.IGNORECASE,
    )
    # A missing separator is safe only when a heading puts the choice on a new line.
    _HEADER_ANSWER_RE = re.compile(
        r"\b(?:final\s+answer|correct\s+(?:answer|choice)|the\s+answer|my\s+answer|answer)\b"
        r"\s*[*_`#]{0,3}\s*"
        r"(?:(?:is\b\s*:?|:|=)\s*[*_`\"'“”‘’]{0,3}\s*[\[(\"'“”‘’]?"
        r"|[ \t]*\r?\n\s*[*_`\"'“”‘’]{0,3}\s*[\[(\"'“”‘’]?)"
        r"([A-D])\b",
        re.IGNORECASE,
    )
    _ANSWER_RE = re.compile(
        r"(?:\b(?:final\s+answer|correct\s+(?:answer|choice)|the\s+answer|my\s+answer|answer)\b"
        r"\s*(?:is\b\s*:?|:|=)\s*[\[(]?([A-D])\b"
        r"|\bI\s+(?:choose|select|pick)\s*[\[(]?([A-D])\b"
        r"|\b(?:I(?:'ll|\s+will)?\s+)?go(?:ing)?\s+with\s*[\[(]?([A-D])\b"
        r"|\b([A-D])\s+is\s+(?:the\s+)?correct(?:\s+(?:answer|choice))?\b)",
        re.IGNORECASE,
    )
    _NEGATED_CORRECTION_RE = re.compile(
        r"\bnot\s+[A-D]\b[\s\S]{0,40}?\b"
        r"(?:it(?:['’]s|\s+is)|(?:the\s+)?answer\s+is|rather|instead)\s*:?\s*([A-D])\b",
        re.IGNORECASE,
    )
    _REJECTED_THEN_AFFIRMED_RE = re.compile(
        r"\b[A-D]\s+(?:is\s+(?:not\s+(?:correct|right)|wrong|incorrect)"
        r"|isn['’]t\s+(?:correct|right))\b"
        r"[\s\S]{0,80}?"
        r"\b([A-D])\s+is(?:\s+(?:right|correct)\b|\s*(?=[.!?,;]|$))",
        re.IGNORECASE,
    )
    _RESULTATIVE_ANSWER_RE = re.compile(
        r"\b(?:making|leav(?:ing|es))\s+([A-D])\s+(?:as\s+)?the\s+correct\s+(?:answer|choice)\b",
        re.IGNORECASE,
    )
    _LEADING_MARKED_RE = re.compile(r"^\s*([A-D])\s*[.):]", re.IGNORECASE)
    # Same-line prose after A is ambiguous with the English article.
    _LEADING_LINE_RE = re.compile(r"^\s*([A-D])\s*(?:\r?\n|$)")
    _LEADING_LINE_CONTINUATION_RE = re.compile(r"^\s*([B-D])\s+")
    _LEADING_ARTICLE_A_RE = re.compile(
        r"^\s*(A)\s+(?=(?:[\"'“”‘’(\[]\s*)?(?:[^\W_]{2}|[^\W_]-[^\W_]))",
    )

    @staticmethod
    def load_questions(path: Path = MCQ_DATA_PATH) -> list[dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def build_prompt(question: dict) -> str:
        choices_text = "\n".join(f"{letter}. {text}" for letter, text in question["choices"].items())
        return (
            f"{question['prompt']}\n\n{choices_text}\n\n"
            "Respond with only the letter of the correct answer."
        )

    @staticmethod
    def parse_answer(response_text: str, valid_choices,
                     allow_unstructured_fallback: bool = True) -> str | None:
        """Extract a structurally stated choice from free-form text."""
        if not response_text:
            return None
        valid = {c.upper() for c in valid_choices}

        stripped = response_text.strip().strip(".()[]:*").strip()
        if len(stripped) == 1 and stripped.upper() in valid:
            return stripped.upper()

        candidates = []
        for pattern in (
            MCQBenchmark._BOXED_RE,
            MCQBenchmark._BOXED_TEXT_RE,
            MCQBenchmark._DECORATED_ANSWER_RE,
            MCQBenchmark._HEADER_ANSWER_RE,
            MCQBenchmark._JSON_ANSWER_RE,
        ):
            candidates.extend(
                (match.start(), match.group(1).upper())
                for match in pattern.finditer(response_text)
                if match.group(1).upper() in valid
            )
        candidates.extend(
            (match.start(), match.group(2).upper())
            for match in MCQBenchmark._TAGGED_RE.finditer(response_text)
            if match.group(2).upper() in valid
        )
        for match in MCQBenchmark._ANSWER_RE.finditer(response_text):
            letter = next(group for group in match.groups() if group is not None).upper()
            if letter in valid:
                candidates.append((match.start(), letter))
        candidates.extend(
            (match.start(), match.group(1).upper())
            for match in MCQBenchmark._NEGATED_CORRECTION_RE.finditer(response_text)
            if match.group(1).upper() in valid
        )
        candidates.extend(
            (match.start(1), match.group(1).upper())
            for match in MCQBenchmark._REJECTED_THEN_AFFIRMED_RE.finditer(response_text)
            if match.group(1).upper() in valid
        )
        candidates.extend(
            (match.start(1), match.group(1).upper())
            for match in MCQBenchmark._RESULTATIVE_ANSWER_RE.finditer(response_text)
            if match.group(1).upper() in valid
        )
        if candidates:
            return max(candidates, key=lambda candidate: candidate[0])[1]

        for pattern in (
            MCQBenchmark._LEADING_MARKED_RE,
            MCQBenchmark._LEADING_LINE_RE,
            MCQBenchmark._LEADING_LINE_CONTINUATION_RE,
        ):
            match = pattern.search(response_text)
            if match and match.group(1).upper() in valid:
                return match.group(1).upper()

        if not allow_unstructured_fallback:
            return None

        leading_article = MCQBenchmark._LEADING_ARTICLE_A_RE.match(response_text)
        found = {
            match.group(1)
            for match in MCQBenchmark._LETTER_RE.finditer(response_text)
            if match.group(1) in valid
            and not (match.group(1) == "A" and leading_article and match.start() == leading_article.start(1))
        }
        return found.pop() if len(found) == 1 else None

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[str | None, str, bool]:
        prompt = MCQBenchmark.build_prompt(question)
        _, _, _, _, response_text, budget_nudged = engine.chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_ctx=config.ACCURACY_CONTEXT,
            num_predict=MCQBenchmark.MCQ_NUM_PREDICT,
            check_loop=True,
            token_budget=config.ACC_TOKEN_BUDGET,
        )
        return (
            MCQBenchmark.parse_answer(response_text, question["choices"].keys()),
            response_text,
            budget_nudged,
        )

    @staticmethod
    def score(questions: list[dict], answers: dict) -> dict:
        """Tally correct/total overall and per category from a {question_id:
        given_letter_or_None} map. Pure, so it's directly testable."""
        by_category: dict[str, dict] = {}
        incorrect = []
        all_results = []
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
