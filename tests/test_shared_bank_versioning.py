from shared import Shared


# ── file_hash ──

def test_file_hash_is_stable_for_same_content(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text('[{"id": "q1"}]')
    assert Shared.file_hash(path) == Shared.file_hash(path)


def test_file_hash_differs_for_different_content(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('[{"id": "q1"}]')
    b.write_text('[{"id": "q1"}, {"id": "q2"}]')
    assert Shared.file_hash(a) != Shared.file_hash(b)


def test_file_hash_is_short_hex():
    # Just a sanity check on the format, not a specific digest.
    import re
    h = Shared.file_hash(__file__)
    assert re.fullmatch(r"[0-9a-f]{12}", h)


# ── stratified_sample ──

def _bank():
    return (
        [{"id": f"a{i}", "category": "alpha"} for i in range(5)]
        + [{"id": f"b{i}", "category": "beta"} for i in range(3)]
        + [{"id": f"c{i}", "category": "gamma"} for i in range(1)]
    )


def test_stratified_sample_returns_everything_when_n_meets_or_exceeds_total():
    bank = _bank()
    assert Shared.stratified_sample(bank, len(bank)) == bank
    assert Shared.stratified_sample(bank, len(bank) + 5) == bank


def test_stratified_sample_returns_exactly_n():
    bank = _bank()
    sample = Shared.stratified_sample(bank, 4)
    assert len(sample) == 4


def test_stratified_sample_touches_every_category_when_n_covers_them():
    bank = _bank()
    sample = Shared.stratified_sample(bank, 3)  # one per category, 3 categories
    assert {q["category"] for q in sample} == {"alpha", "beta", "gamma"}


def test_stratified_sample_is_deterministic_across_calls():
    bank = _bank()
    first = [q["id"] for q in Shared.stratified_sample(bank, 4)]
    second = [q["id"] for q in Shared.stratified_sample(bank, 4)]
    assert first == second


def test_stratified_sample_no_duplicate_ids():
    bank = _bank()
    sample = Shared.stratified_sample(bank, 6)
    ids = [q["id"] for q in sample]
    assert len(ids) == len(set(ids))


# ── check_crash_cache / record_crash bank-hash gating ──

def test_check_crash_cache_ignores_stale_entry_from_different_bank(tmp_path):
    path = tmp_path / "crash.json"
    cache = {"some-tag": {"crashed_at": "2026-01-01T00:00:00", "bank_hash": "old-hash"}}
    entry = Shared.check_crash_cache("some-tag", "Some Model", cache, path, expected_bank_hash="new-hash")
    assert entry is None


def test_check_crash_cache_honors_entry_matching_current_bank(tmp_path):
    path = tmp_path / "crash.json"
    cache = {"some-tag": {"crashed_at": "2026-01-01T00:00:00", "bank_hash": "current-hash"}}
    entry = Shared.check_crash_cache("some-tag", "Some Model", cache, path, expected_bank_hash="current-hash")
    assert entry is not None
    assert entry["skipped"] is True


def test_check_crash_cache_without_expected_hash_ignores_bank_field(tmp_path):
    # Non-bank-aware callers (llm/conv/emb) never pass expected_bank_hash —
    # a record with a bank_hash field must still be honored for them.
    path = tmp_path / "crash.json"
    cache = {"some-tag": {"crashed_at": "2026-01-01T00:00:00", "bank_hash": "whatever"}}
    entry = Shared.check_crash_cache("some-tag", "Some Model", cache, path)
    assert entry is not None


def test_record_crash_stores_extra_fields(tmp_path):
    path = tmp_path / "crash.json"
    cache = {}
    Shared.record_crash("some-tag", cache, path, "answering q1", extra={"bank_hash": "abc123"})
    assert cache["some-tag"]["bank_hash"] == "abc123"
    assert "crashed_at" in cache["some-tag"]


def test_record_crash_without_extra_omits_bank_hash(tmp_path):
    path = tmp_path / "crash.json"
    cache = {}
    Shared.record_crash("some-tag", cache, path, "warming up")
    assert "bank_hash" not in cache["some-tag"]
