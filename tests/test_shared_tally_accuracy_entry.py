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
