import copy
import json

import pytest

import config
from mcq_benchmark import MCQBenchmark
from reasoning_benchmark import ReasoningBenchmark


CHOICES = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}


def valid_bank():
    return {
        "schema_version": 1,
        "bank_version": "1.0",
        "title": "Reasoning",
        "description": "A test bank.",
        "research_basis": [{
            "name": "Source",
            "url": "https://example.com/source",
            "contribution": "Inspired the categories.",
        }],
        "categories": [{
            "id": "logic",
            "label": "Logic",
            "description": "Deductive reasoning.",
        }],
        "questions": [{
            "id": "reason_001",
            "category": "logic",
            "difficulty": "hard",
            "prompt": "Which choice follows?",
            "choices": copy.deepcopy(CHOICES),
            "answer": "B",
            "rationale": "The premises imply beta.",
            "skills": ["deduction", "negation"],
            "provenance": "original",
        }],
    }


def test_build_prompt_includes_only_model_facing_question_content():
    question = valid_bank()["questions"][0]
    prompt = ReasoningBenchmark.build_prompt(question)

    assert question["prompt"] in prompt
    assert all(f"{letter}. {text}" in prompt for letter, text in CHOICES.items())
    assert question["answer"] not in prompt.splitlines()[-1]
    assert question["rationale"] not in prompt
    assert "only the letter" in prompt


@pytest.mark.parametrize(("response", "expected"), [
    ("B", "B"),
    ("(b).", "B"),
    ("Final answer: D", "D"),
    ("The correct choice is C.", "C"),
    ("I choose A", "A"),
    ("I'll go with B", "B"),
    ("C is the correct answer", "C"),
    (r"\boxed{D}", "D"),
    (r"\boxed{\text{A}}", "A"),
    (r"\boxed{\mathbf{B}}", "B"),
    (r"\boxed{\mathrm{C}}", "C"),
    ("<answer>b</answer>", "B"),
    ("<final_answer>D</final_answer>", "D"),
    ('{"answer": "A"}', "A"),
    ("Final answer: **C**", "C"),
    ("D. Because the fourth assignment is consistent.", "D"),
    ("A\nThe first ordering is the only valid one.", "A"),
    ("**Final Answer**: A", "A"),
    ("**Answer**: A", "A"),
    ("Final Answer\nA", "A"),
    ("### Final Answer\nA", "A"),
    ('The answer is: "A"', "A"),
    ("The answer is: 'A'", "A"),
    ("The answer is: “A”", "A"),
    ("The answer is: ‘A’", "A"),
])
def test_parse_answer_accepts_structurally_stated_choices(response, expected):
    assert ReasoningBenchmark.parse_answer(response, CHOICES) == expected


@pytest.mark.parametrize("response", [
    "Considering C carefully, the evidence supports C.",
    "Option A fails while option B remains possible.",
    "Both C and A may work.",
    "The words contain no final selection.",
    r"$\boxed{452}$",
    "<answer>C",
    "The answer is not entirely clear, though Alice thinks A might work, but Bob prefers D.",
    "A reasonable answer A is one option among many, though I lean towards D.",
    "The sample answer A shown in the textbook differs from mine, which is D.",
    "A wise man once said the answer would depend on context.",
    "A number of factors point to a plausible reading, none conclusive.",
    "A Number of factors make this uncertain.",
    "A JSON object can represent the result.",
    'A "wise choice" depends on context.',
    "A URL by itself does not settle the question.",
    "A 5-step argument is still inconclusive.",
    "A X-ray result would not settle this puzzle.",
    "",
    None,
])
def test_parse_answer_rejects_ambiguous_or_unstructured_mentions(response):
    assert ReasoningBenchmark.parse_answer(response, CHOICES) is None


def test_reasoning_strict_mode_does_not_change_mcq_fallback_behavior():
    response = "Considering C carefully, the evidence still supports C."
    assert MCQBenchmark.parse_answer(response, CHOICES) == "C"
    assert ReasoningBenchmark.parse_answer(response, CHOICES) is None


@pytest.mark.parametrize(("response", "expected"), [
    ("The answer is A. Rechecking, the final answer is D.", "D"),
    ("The answer is not A, it's B.", "B"),
    ("The answer is not A, it’s B.", "B"),
    ("A is wrong. C is correct.", "C"),
    ("I choose B. Rechecking leaves D as the correct choice.", "D"),
    ("The answer is C, not D.", "C"),
])
def test_parse_answer_handles_explicit_corrections_without_option_leakage(response, expected):
    assert ReasoningBenchmark.parse_answer(response, CHOICES) == expected


def test_parse_answer_ignores_explicit_choice_outside_valid_set():
    assert ReasoningBenchmark.parse_answer("Final answer: D", {"A", "B", "C"}) is None


def test_load_questions_validates_the_real_bank():
    questions = ReasoningBenchmark.load_questions()
    assert len(questions) == 60
    assert questions[0]["id"] == "reason_001"
    assert questions[-1]["id"] == "reason_060"


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda bank: bank.update(extra=True), "top-level field set"),
    (lambda bank: bank.update(schema_version=2), "schema_version"),
    (lambda bank: bank.update(bank_version=" "), "bank_version"),
    (lambda bank: bank.update(title=""), "title"),
    (lambda bank: bank.update(description=None), "description"),
    (lambda bank: bank.update(research_basis=[]), "research basis"),
    (lambda bank: bank["research_basis"][0].update(extra="x"), "research source"),
    (lambda bank: bank["research_basis"][0].update(name=""), "non-empty"),
    (lambda bank: bank["research_basis"][0].update(url="http://example.com"), "HTTPS"),
    (lambda bank: bank.update(categories=[]), "define categories"),
    (lambda bank: bank["categories"][0].update(extra="x"), "id, label, and description"),
    (lambda bank: bank["categories"][0].update(label=" "), "non-empty strings"),
    (lambda bank: bank["categories"][0].update(id=" logic"), "surrounding whitespace"),
    (lambda bank: bank["categories"].append(copy.deepcopy(bank["categories"][0])), "unique"),
    (lambda bank: bank.update(questions=[]), "contain questions"),
    (lambda bank: bank["questions"][0].update(extra="x"), "field set"),
    (lambda bank: bank["questions"][0].update(id=" "), "invalid ID"),
    (lambda bank: bank["questions"].append(copy.deepcopy(bank["questions"][0])), "duplicate"),
    (lambda bank: bank["questions"][0].update(category="missing"), "unknown category"),
    (lambda bank: bank["questions"][0].update(difficulty="extreme"), "difficulty"),
    (lambda bank: bank["questions"][0].update(prompt=" "), "empty prompt"),
    (lambda bank: bank["questions"][0].update(choices={"A": "a"}), "exactly choices"),
    (lambda bank: bank["questions"][0]["choices"].update(A=" "), "empty choice"),
    (lambda bank: bank["questions"][0]["choices"].update(A=" BETA "), "distinct"),
    (lambda bank: bank["questions"][0].update(answer="E"), "valid choice"),
    (lambda bank: bank["questions"][0].update(rationale=""), "empty rationale"),
    (lambda bank: bank["questions"][0].update(skills=[]), "skill tags"),
    (lambda bank: bank["questions"][0].update(skills=[None]), "invalid skill"),
    (lambda bank: bank["questions"][0].update(skills=["Deduction", " deduction "]), "unique"),
    (lambda bank: bank["questions"][0].update(provenance="adapted"), "original"),
])
def test_validate_bank_rejects_malformed_content(mutate, message):
    bank = valid_bank()
    mutate(bank)
    with pytest.raises(ValueError, match=message):
        ReasoningBenchmark.validate_bank(bank)


@pytest.mark.parametrize("bank", [None, [], "bank"])
def test_validate_bank_requires_an_object(bank):
    with pytest.raises(ValueError, match="JSON object"):
        ReasoningBenchmark.validate_bank(bank)


def test_validate_bank_rejects_unpopulated_category():
    bank = valid_bank()
    bank["categories"].append({
        "id": "spatial", "label": "Spatial", "description": "Spatial reasoning.",
    })
    with pytest.raises(ValueError, match="categories without questions.*spatial"):
        ReasoningBenchmark.validate_bank(bank)


def test_load_questions_wraps_invalid_json_with_bank_path(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{")
    with pytest.raises(ValueError, match=r"could not read reasoning bank.*broken\.json"):
        ReasoningBenchmark.load_questions(path)


def test_load_questions_wraps_missing_file_with_bank_path(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match=r"could not read reasoning bank.*missing\.json"):
        ReasoningBenchmark.load_questions(path)


class FakeEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return 0.0, 0.0, 0.0, 0.0, self.response


def test_ask_uses_accuracy_limits_loop_detection_and_strict_parser(monkeypatch):
    monkeypatch.setattr(config, "ACC_TIMEOUT", 17)
    monkeypatch.setattr(config, "ACCURACY_CONTEXT", 8192)
    engine = FakeEngine("Considering C carefully, the evidence supports C.")
    question = valid_bank()["questions"][0]

    parsed, raw = ReasoningBenchmark._ask(engine, "model", question)

    assert parsed is None
    assert raw == engine.response
    args, kwargs = engine.calls[0]
    assert args[0] == "model"
    assert args[1] == [{"role": "user", "content": ReasoningBenchmark.build_prompt(question)}]
    assert kwargs == {
        "timeout": 17,
        "num_ctx": 8192,
        "num_predict": -1,
        "check_loop": True,
    }


def scoring_questions():
    first = valid_bank()["questions"][0]
    second = copy.deepcopy(first)
    second.update(id="reason_002", category="spatial", difficulty="very_hard", answer="C")
    third = copy.deepcopy(first)
    third.update(id="reason_003", difficulty="very_hard", answer="A")
    return [first, second, third]


def test_score_reports_category_and_difficulty_breakdowns():
    result = ReasoningBenchmark.score(
        scoring_questions(), {"reason_001": "B", "reason_002": "D"},
    )

    assert result["correct"] == 1
    assert result["total"] == 3
    assert result["answered"] == 2
    assert result["accuracy_pct"] == 33.3
    assert result["by_category"] == {
        "logic": {"correct": 1, "total": 2, "accuracy_pct": 50.0},
        "spatial": {"correct": 0, "total": 1, "accuracy_pct": 0.0},
    }
    assert result["by_difficulty"] == {
        "hard": {"correct": 1, "total": 1, "accuracy_pct": 100.0},
        "very_hard": {"correct": 0, "total": 2, "accuracy_pct": 0.0},
    }
    assert [entry["id"] for entry in result["incorrect"]] == ["reason_002", "reason_003"]
    assert result["all"][0] == {
        "id": "reason_001", "category": "logic", "difficulty": "hard",
        "given": "B", "expected": "B", "correct": True,
    }


def test_score_handles_an_empty_sample():
    assert ReasoningBenchmark.score([], {}) == {
        "correct": 0,
        "total": 0,
        "answered": 0,
        "accuracy_pct": 0.0,
        "by_category": {},
        "by_difficulty": {},
        "incorrect": [],
        "all": [],
    }
