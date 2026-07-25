from shared import Shared


def test_tally_correct_entry_updates_category_and_all_but_not_incorrect():
    cat = {"correct": 0, "total": 1}
    all_results, incorrect = [], []
    entry = {"id": "q1", "category": "science", "given": "B", "expected": "B"}

    is_correct = Shared.tally_accuracy_entry(entry, True, cat, all_results, incorrect)

    assert is_correct is True
    assert cat == {"correct": 1, "total": 1}
    assert incorrect == []
    assert all_results == [{"id": "q1", "category": "science", "given": "B", "expected": "B", "correct": True}]


def test_tally_incorrect_entry_appends_to_incorrect_without_correct_key():
    cat = {"correct": 0, "total": 1}
    all_results, incorrect = [], []
    entry = {"id": "q2", "category": "science", "given": "C", "expected": "A"}

    is_correct = Shared.tally_accuracy_entry(entry, False, cat, all_results, incorrect)

    assert is_correct is False
    assert cat == {"correct": 0, "total": 1}
    # incorrect keeps the bare entry (no "correct" key), matching the main results JSON's existing shape.
    assert incorrect == [{"id": "q2", "category": "science", "given": "C", "expected": "A"}]
    assert all_results == [{"id": "q2", "category": "science", "given": "C", "expected": "A", "correct": False}]


def test_tally_does_not_mutate_the_original_entry_dict():
    cat = {"correct": 0, "total": 1}
    entry = {"id": "q1", "category": "science"}

    Shared.tally_accuracy_entry(entry, True, cat, [], [])

    assert entry == {"id": "q1", "category": "science"}  # unchanged — "correct" only added to the all-list copy


def test_finalize_accuracy_score_fills_category_pct_and_assembles_result():
    by_category = {"science": {"correct": 1, "total": 2}, "math": {"correct": 0, "total": 0}}
    incorrect = [{"id": "q2"}]
    all_results = [{"id": "q1", "correct": True}, {"id": "q2", "correct": False}]

    result = Shared.finalize_accuracy_score(2, 1, 2, by_category, incorrect, all_results)

    assert result["by_category"]["science"]["accuracy_pct"] == 50.0
    assert result["by_category"]["math"]["accuracy_pct"] == 0.0  # zero-total category doesn't divide by zero
    assert result["accuracy_pct"] == 50.0
    assert result["incorrect"] == incorrect
    assert result["all"] == all_results
    assert "by_difficulty" not in result


def test_finalize_accuracy_score_zero_total_avoids_division_by_zero():
    result = Shared.finalize_accuracy_score(0, 0, 0, {}, [], [])

    assert result["accuracy_pct"] == 0.0


def test_finalize_accuracy_score_includes_extra_breakdowns():
    by_category = {"logic": {"correct": 1, "total": 1}}
    by_difficulty = {"hard": {"correct": 1, "total": 2}}

    result = Shared.finalize_accuracy_score(1, 1, 1, by_category, [], [],
                                             extra={"by_difficulty": by_difficulty})

    assert result["by_difficulty"]["hard"]["accuracy_pct"] == 50.0
    assert result["by_category"]["logic"]["accuracy_pct"] == 100.0
