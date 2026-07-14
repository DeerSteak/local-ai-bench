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

    # Records models that crashed Ollama's runner repeatedly (deterministically,
    # not a transient blip) so future runs don't waste time rediscovering the
    # same crash. Delete this file to retry a skipped model.
    MATH_CRASH_CACHE = Path(".math_crash_cache.json")

    # Unbounded (-1): a fixed token cap risks truncating a reasoning model's
    # answer. The wall-clock timeout in Shared.ollama_chat is the real bound.
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
    def _ask(tag: str, question: dict) -> tuple[float | None, str]:
        prompt = MathBenchmark.build_prompt(question)
        _, _, _, _, response_text = Shared.ollama_chat(
            tag, [{"role": "user", "content": prompt}],
            timeout=config.ACC_TIMEOUT, num_predict=MathBenchmark.MATH_NUM_PREDICT,
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

    def run(self, models, questions=None, warmup_runs=config.WARMUP_RUNS, save_fn=None,
            answers_path: Path | None = None):  # pragma: no cover — orchestrates real Ollama runs
        results = {}
        answers_out: dict = {}
        questions = questions if questions is not None else MathBenchmark.load_questions()

        if not Shared.ollama_available():
            Shared.err("Ollama server not reachable — skipping math benchmark")
            Shared.err("Start with: ollama serve")
            return results

        crash_cache = Shared.load_crash_cache(MathBenchmark.MATH_CRASH_CACHE)
        bank_hash = Shared.file_hash(MathBenchmark.MATH_DATA_PATH)

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"Math: {label}")

            if not Shared.ollama_reachable_or_abort():
                break

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, MathBenchmark.MATH_CRASH_CACHE,
                                                       expected_bank_hash=bank_hash)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                if not Shared.warmup_model(tag, label, config.CONTEXT_LENGTHS[0], warmup_runs,
                                           crash_cache, MathBenchmark.MATH_CRASH_CACHE,
                                           crash_extra={"bank_hash": bank_hash}):
                    Shared.unload_model(tag)
                    continue

                Shared.log(f"Answering {len(questions)} math questions "
                           f"({config.ACC_TIMEOUT}s timeout each) ...")
                answers: dict[str, float | None] = {}
                raw_responses: dict[str, str] = {}
                timed_out_ids: list[str] = []
                likely_loop_ids: list[str] = []
                stopped_early = None

                for i, q in enumerate(questions):
                    samples, status, partial_text = Shared.run_measured_calls(
                        1, lambda run_i, q=q: MathBenchmark._ask(tag, q), tag, crash_cache,
                        MathBenchmark.MATH_CRASH_CACHE, f"answering {q['id']}",
                        crash_extra={"bank_hash": bank_hash})
                    if samples:
                        given, raw = samples[0]
                    elif status == "timed_out" and partial_text:
                        # Score whatever the model had written before the wall-clock
                        # timeout hit, rather than treating it as a blank — this is
                        # either an answer cut off right at the end or unparseable
                        # (wrong-format) text, not necessarily "no output at all."
                        given, raw = MathBenchmark.parse_answer(partial_text), partial_text
                    else:
                        given, raw = None, ""
                    answers[q["id"]] = given
                    raw_responses[q["id"]] = raw

                    if status == "timed_out":
                        # A single stuck question is scored wrong and the run moves
                        # on — see MCQBenchmark.run for why this replaced abandoning
                        # the rest of the bank on the first timeout.
                        Shared.warn(f"{q['id']} timed out after {config.ACC_TIMEOUT}s — "
                                    "scoring as wrong and continuing")
                        timed_out_ids.append(q["id"])
                        if partial_text and Shared.looks_like_loop(partial_text):
                            Shared.warn(f"{q['id']}: response looks like a generation loop")
                            likely_loop_ids.append(q["id"])
                    if status == "crashed":
                        stopped_early = "crashed"
                        break

                    if (i + 1) % 10 == 0:
                        Shared.log(f"  {i+1}/{len(questions)} answered ...")

                scored = MathBenchmark.score(questions, answers)
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
                if likely_loop_ids:
                    results[short]["likely_loop_count"] = len(likely_loop_ids)
                    results[short]["likely_loop_ids"] = likely_loop_ids
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
