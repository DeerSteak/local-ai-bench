import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def test_reasoning_bank_has_versioned_top_level_and_question_shapes():
    bank = load_json("reasoning_questions.json")
    assert bank["schema_version"] == 1
    assert set(bank) == {
        "schema_version", "bank_version", "title", "description",
        "research_basis", "categories", "questions",
    }

    required = {
        "id", "category", "difficulty", "prompt", "choices", "answer",
        "rationale", "skills", "provenance",
    }
    for question in bank["questions"]:
        assert set(question) == required
        assert set(question["choices"]) == {"A", "B", "C", "D"}
        assert question["answer"] in question["choices"]
        assert len(set(question["choices"].values())) == 4
        assert question["difficulty"] in {"easy", "medium", "hard", "very_hard"}
        assert question["provenance"] == "original"
        assert question["rationale"].strip()
        assert question["skills"] and len(question["skills"]) == len(set(question["skills"]))


def test_reasoning_bank_is_balanced_and_every_category_is_populated():
    bank = load_json("reasoning_questions.json")
    category_ids = [category["id"] for category in bank["categories"]]
    questions = bank["questions"]
    assert len(category_ids) == len(set(category_ids)) == 10
    assert len(questions) == 60
    assert len({question["id"] for question in questions}) == len(questions)
    assert Counter(question["category"] for question in questions) == {
        category_id: 6 for category_id in category_ids
    }
    assert Counter(question["answer"] for question in questions) == {
        "A": 15, "B": 15, "C": 15, "D": 15,
    }
    for category_id in category_ids:
        difficulties = {
            question["difficulty"] for question in questions
            if question["category"] == category_id
        }
        assert difficulties == {"easy", "medium", "hard", "very_hard"}

    final_questions = questions[-20:]
    assert [question["id"] for question in final_questions] == [
        f"reason_{number:03}" for number in range(41, 61)
    ]
    assert all(question["difficulty"] == "very_hard" for question in final_questions)
    assert Counter(question["category"] for question in final_questions) == {
        category_id: 2 for category_id in category_ids
    }


def test_reasoning_bank_research_sources_are_named_https_urls():
    bank = load_json("reasoning_questions.json")
    for source in bank["research_basis"]:
        assert set(source) == {"name", "url", "contribution"}
        assert source["url"].startswith("https://")
        assert source["name"].strip() and source["contribution"].strip()
