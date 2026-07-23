"""Knowledge-light reasoning accuracy benchmark."""

import json
from collections import Counter
from pathlib import Path

import config
from mcq_benchmark import MCQBenchmark
from shared import Shared


class ReasoningBenchmark:
    REASONING_DATA_PATH = config.SCRIPT_DIR / "scripts" / "data" / "reasoning_questions.json"
    REASONING_CRASH_CACHE = Path(".reasoning_crash_cache.json")
    REASONING_NUM_PREDICT = -1
    DIFFICULTIES = {"easy", "medium", "hard", "very_hard"}
    QUESTION_FIELDS = {
        "id", "category", "difficulty", "prompt", "choices", "answer",
        "rationale", "skills", "provenance",
    }
    BANK_FIELDS = {
        "schema_version", "bank_version", "title", "description",
        "research_basis", "categories", "questions",
    }
    RESEARCH_FIELDS = {"name", "url", "contribution"}

    @staticmethod
    def validate_bank(bank: dict) -> list[dict]:
        if not isinstance(bank, dict):
            raise ValueError("reasoning bank must be a JSON object")
        if set(bank) != ReasoningBenchmark.BANK_FIELDS:
            raise ValueError("reasoning bank has an invalid top-level field set")
        if bank.get("schema_version") != 1:
            raise ValueError("reasoning bank schema_version must be 1")
        if not isinstance(bank.get("bank_version"), str) or not bank["bank_version"].strip():
            raise ValueError("reasoning bank must have a bank_version string")
        for field in ("title", "description"):
            if not isinstance(bank.get(field), str) or not bank[field].strip():
                raise ValueError(f"reasoning bank must have a {field} string")

        research_basis = bank.get("research_basis")
        if not isinstance(research_basis, list) or not research_basis:
            raise ValueError("reasoning bank must document its research basis")
        for source in research_basis:
            if not isinstance(source, dict) or set(source) != ReasoningBenchmark.RESEARCH_FIELDS:
                raise ValueError("each reasoning research source has an invalid field set")
            if not all(isinstance(source[key], str) and source[key].strip() for key in source):
                raise ValueError("reasoning research source fields must be non-empty strings")
            if not source["url"].startswith("https://"):
                raise ValueError("reasoning research source URLs must use HTTPS")

        categories = bank.get("categories")
        if not isinstance(categories, list) or not categories:
            raise ValueError("reasoning bank must define categories")
        category_ids = []
        for category in categories:
            if not isinstance(category, dict) or set(category) != {"id", "label", "description"}:
                raise ValueError("each reasoning category must have id, label, and description")
            if not all(isinstance(category[key], str) and category[key].strip() for key in category):
                raise ValueError("reasoning category fields must be non-empty strings")
            if category["id"] != category["id"].strip():
                raise ValueError("reasoning category IDs cannot have surrounding whitespace")
            category_ids.append(category["id"])
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("reasoning category IDs must be unique")

        questions = bank.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError("reasoning bank must contain questions")
        seen_ids = set()
        category_counts = Counter()
        for index, question in enumerate(questions, 1):
            if not isinstance(question, dict) or set(question) != ReasoningBenchmark.QUESTION_FIELDS:
                raise ValueError(f"reasoning question {index} has an invalid field set")
            question_id = question["id"]
            if (not isinstance(question_id, str) or not question_id.strip()
                    or question_id != question_id.strip()):
                raise ValueError(f"reasoning question {index} has an invalid ID")
            if question_id in seen_ids:
                raise ValueError(f"duplicate reasoning question ID: {question_id}")
            seen_ids.add(question_id)
            if not isinstance(question["category"], str) or question["category"] not in category_ids:
                raise ValueError(f"{question_id} uses an unknown category")
            category_counts[question["category"]] += 1
            if (not isinstance(question["difficulty"], str)
                    or question["difficulty"] not in ReasoningBenchmark.DIFFICULTIES):
                raise ValueError(f"{question_id} has an invalid difficulty")
            if not isinstance(question["prompt"], str) or not question["prompt"].strip():
                raise ValueError(f"{question_id} has an empty prompt")
            choices = question["choices"]
            if not isinstance(choices, dict) or set(choices) != {"A", "B", "C", "D"}:
                raise ValueError(f"{question_id} must have exactly choices A-D")
            if not all(isinstance(value, str) and value.strip() for value in choices.values()):
                raise ValueError(f"{question_id} has an empty choice")
            normalized_choices = {value.strip().casefold() for value in choices.values()}
            if len(normalized_choices) != 4:
                raise ValueError(f"{question_id} choices must be distinct")
            if not isinstance(question["answer"], str) or question["answer"] not in choices:
                raise ValueError(f"{question_id} answer is not a valid choice")
            if not isinstance(question["rationale"], str) or not question["rationale"].strip():
                raise ValueError(f"{question_id} has an empty rationale")
            skills = question["skills"]
            if not isinstance(skills, list) or not skills:
                raise ValueError(f"{question_id} must have skill tags")
            if not all(isinstance(skill, str) and skill.strip() for skill in skills):
                raise ValueError(f"{question_id} has an invalid skill tag")
            normalized_skills = {skill.strip().casefold() for skill in skills}
            if len(skills) != len(normalized_skills):
                raise ValueError(f"{question_id} must have unique skill tags")
            if question["provenance"] != "original":
                raise ValueError(f"{question_id} provenance must be original")

        missing_categories = [category for category in category_ids if not category_counts[category]]
        if missing_categories:
            raise ValueError(f"reasoning categories without questions: {missing_categories}")
        return questions

    @staticmethod
    def load_questions(path: Path = REASONING_DATA_PATH) -> list[dict]:
        try:
            bank = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read reasoning bank {path}: {exc}") from exc
        return ReasoningBenchmark.validate_bank(bank)

    @staticmethod
    def build_prompt(question: dict) -> str:
        choices_text = "\n".join(
            f"{letter}. {text}" for letter, text in question["choices"].items()
        )
        return (
            f"{question['prompt']}\n\n{choices_text}\n\n"
            "Solve the problem carefully. Respond with only the letter of the correct answer."
        )

    @staticmethod
    def parse_answer(response_text: str, valid_choices) -> str | None:
        return MCQBenchmark.parse_answer(
            response_text, valid_choices, allow_unstructured_fallback=False,
        )

    @staticmethod
    def _ask(engine, tag: str, question: dict) -> tuple[str | None, str, bool]:
        _, _, _, _, response_text, budget_nudged = engine.chat(
            tag,
            [{"role": "user", "content": ReasoningBenchmark.build_prompt(question)}],
            timeout=config.ACC_TIMEOUT,
            num_ctx=config.ACCURACY_CONTEXT,
            num_predict=ReasoningBenchmark.REASONING_NUM_PREDICT,
            check_loop=True,
            token_budget=config.ACC_TOKEN_BUDGET,
        )
        return ReasoningBenchmark.parse_answer(
            response_text, question["choices"].keys(),
        ), response_text, budget_nudged

    @staticmethod
    def score(questions: list[dict], answers: dict) -> dict:
        by_category: dict[str, dict] = {}
        by_difficulty: dict[str, dict] = {}
        incorrect = []
        all_results = []
        correct = 0
        answered = 0

        for question in questions:
            question_id = question["id"]
            category = question["category"]
            difficulty = question["difficulty"]
            expected = question["answer"]
            given = answers.get(question_id)
            category_score = by_category.setdefault(category, {"correct": 0, "total": 0})
            difficulty_score = by_difficulty.setdefault(
                difficulty, {"correct": 0, "total": 0},
            )
            category_score["total"] += 1
            difficulty_score["total"] += 1
            if given is not None:
                answered += 1
            is_correct = given == expected
            entry = {
                "id": question_id,
                "category": category,
                "difficulty": difficulty,
                "given": given,
                "expected": expected,
            }
            if Shared.tally_accuracy_entry(
                entry, is_correct, category_score, all_results, incorrect,
            ):
                correct += 1
                difficulty_score["correct"] += 1

        for breakdown in (by_category, by_difficulty):
            for score in breakdown.values():
                score["accuracy_pct"] = (
                    round(100 * score["correct"] / score["total"], 1)
                    if score["total"] else 0.0
                )

        total = len(questions)
        return {
            "correct": correct,
            "total": total,
            "answered": answered,
            "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "incorrect": incorrect,
            "all": all_results,
        }

    def run(self, engine, models, questions=None, warmup_runs=config.WARMUP_RUNS,
            save_fn=None, answers_path: Path | None = None
            ):  # pragma: no cover — orchestrates real engine runs
        questions = questions if questions is not None else ReasoningBenchmark.load_questions()
        return Shared.run_accuracy_benchmark(
            section_label="Reasoning",
            skip_label="reasoning",
            question_noun="reasoning questions",
            data_path=ReasoningBenchmark.REASONING_DATA_PATH,
            crash_cache_path=ReasoningBenchmark.REASONING_CRASH_CACHE,
            models=models,
            questions=questions,
            warmup_runs=warmup_runs,
            engine=engine,
            ask_fn=lambda tag, question: ReasoningBenchmark._ask(engine, tag, question),
            rescore_partial_fn=lambda question, text: ReasoningBenchmark.parse_answer(
                text, question["choices"].keys(),
            ),
            score_fn=ReasoningBenchmark.score,
            save_fn=save_fn,
            answers_path=answers_path,
        )
