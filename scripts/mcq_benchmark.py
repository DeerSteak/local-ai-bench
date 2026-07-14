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

    # Records models that crashed Ollama's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    MCQ_CRASH_CACHE = Path(".mcq_crash_cache.json")

    # Unbounded (-1): a fixed token cap risks truncating a reasoning model's
    # answer. The wall-clock timeout in Shared.ollama_chat is the real bound.
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

        Takes the *last* standalone valid letter, not the first — MCQ_NUM_PREDICT
        is unbounded specifically so a reasoning model has room to think before
        answering, and that reasoning routinely walks through several options by
        letter ("A is wrong because... B is wrong because... so the answer is
        C") before landing on a final choice. Taking the first letter would grab
        a rejected option instead of the model's actual answer; a stray letter
        that isn't a valid choice for this question ("A" in "As an AI...") is
        skipped regardless of position. Mirrors MathBenchmark.parse_answer's
        same last-not-first reasoning for the same reason.
        """
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
    def _ask(tag: str, question: dict) -> tuple[str | None, str]:
        prompt = MCQBenchmark.build_prompt(question)
        _, _, _, _, response_text = Shared.ollama_chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_predict=MCQBenchmark.MCQ_NUM_PREDICT,
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

    def run(self, models, questions=None, warmup_runs=config.WARMUP_RUNS, save_fn=None,
            answers_path: Path | None = None):  # pragma: no cover — orchestrates real Ollama runs
        results = {}
        answers_out: dict = {}
        questions = questions if questions is not None else MCQBenchmark.load_questions()

        if not Shared.ollama_available():
            Shared.err("Ollama server not reachable — skipping MCQ benchmark")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(MCQBenchmark.MCQ_CRASH_CACHE)
        bank_hash = Shared.file_hash(MCQBenchmark.MCQ_DATA_PATH)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"MCQ: {label}")

            if not Shared.ollama_reachable_or_abort():
                break

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, MCQBenchmark.MCQ_CRASH_CACHE,
                                                       expected_bank_hash=bank_hash)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                if not Shared.warmup_model(tag, label, config.CONTEXT_LENGTHS[0], warmup_runs,
                                           crash_cache, MCQBenchmark.MCQ_CRASH_CACHE,
                                           crash_extra={"bank_hash": bank_hash}):
                    Shared.unload_model(tag)
                    continue

                Shared.log(f"Answering {len(questions)} MCQ questions "
                           f"({config.ACC_TIMEOUT}s timeout each) ...")
                answers: dict[str, str | None] = {}
                raw_responses: dict[str, str] = {}
                timed_out_ids: list[str] = []
                stopped_early = None

                for i, q in enumerate(questions):
                    samples, status, partial_text = Shared.run_measured_calls(
                        1, lambda run_i, q=q: MCQBenchmark._ask(tag, q), tag, crash_cache,
                        MCQBenchmark.MCQ_CRASH_CACHE, f"answering {q['id']}",
                        crash_extra={"bank_hash": bank_hash})
                    if samples:
                        given, raw = samples[0]
                    elif status == "timed_out" and partial_text:
                        # The model had already started answering when the wall-clock
                        # timeout hit. Score whatever it wrote instead of a blank —
                        # this is either a genuinely correct/incorrect answer cut off
                        # right at the end, or unparseable (wrong-format) text, not
                        # necessarily "the model produced nothing."
                        given, raw = MCQBenchmark.parse_answer(partial_text, q["choices"].keys()), partial_text
                    else:
                        given, raw = None, ""
                    answers[q["id"]] = given
                    raw_responses[q["id"]] = raw

                    if status == "timed_out":
                        # A single stuck question is scored wrong and the run moves
                        # on — with ACC_TIMEOUT this short, a model that reliably
                        # gets stuck could otherwise rack up timeouts on a sizeable
                        # fraction of the bank, but that's still cheaper than the
                        # old behavior of abandoning the rest of the bank outright
                        # (and would incorrectly zero out everything after one bad
                        # question for a model that's merely slow, not stuck).
                        timed_out_ids.append(q["id"])
                    if status == "crashed":
                        stopped_early = "crashed"
                        break

                    if (i + 1) % 10 == 0:
                        Shared.log(f"  {i+1}/{len(questions)} answered ...")

                scored = MCQBenchmark.score(questions, answers)
                answers_out[short] = {
                    "label": label,
                    "incorrect": [
                        {**entry, "raw_response": raw_responses.get(entry["id"], "")}
                        for entry in scored["incorrect"]
                    ],
                }
                results[short] = {"label": label, **scored}

                if timed_out_ids:
                    results[short]["timed_out_count"] = len(timed_out_ids)
                    results[short]["timed_out_ids"] = timed_out_ids
                if stopped_early == "crashed":
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
