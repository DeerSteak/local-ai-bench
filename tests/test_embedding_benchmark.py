from scripts.workloads.embedding_benchmark import EmbeddingBenchmark
from scripts.runtime.engines.base import EmbeddingMeasurement


def _write_doc(tmp_path, text):
    path = tmp_path / "doc.txt"
    path.write_text(text)
    return path


def test_short_paragraphs_become_single_chunks(tmp_path):
    doc = _write_doc(tmp_path, "This is a short paragraph with enough words in it.\n\nAnd a second one here too.")
    chunks = EmbeddingBenchmark.chunk_document(doc, max_words=150, min_words=6)
    assert len(chunks) == 2
    assert "short paragraph" in chunks[0]


def test_paragraphs_below_min_words_are_dropped(tmp_path):
    doc = _write_doc(tmp_path, "Too short.\n\nThis paragraph on the other hand has plenty of words in it to keep.")
    chunks = EmbeddingBenchmark.chunk_document(doc, max_words=150, min_words=6)
    assert len(chunks) == 1
    assert "plenty of words" in chunks[0]


def test_no_chunk_ever_exceeds_max_words(tmp_path):
    # One long paragraph with normal sentence punctuation.
    sentence = "The quick brown fox jumps over the lazy dog again and again. "
    doc = _write_doc(tmp_path, sentence * 40)
    chunks = EmbeddingBenchmark.chunk_document(doc, max_words=20, min_words=6)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 20


def test_oversized_content_with_no_punctuation_is_hard_split(tmp_path):
    # No sentence-ending punctuation anywhere, e.g. a code block or table.
    words = " ".join(f"word{i}" for i in range(100))
    doc = _write_doc(tmp_path, words)
    chunks = EmbeddingBenchmark.chunk_document(doc, max_words=10, min_words=6)
    assert len(chunks) == 10
    for chunk in chunks:
        assert len(chunk.split()) <= 10
    # No words lost or reordered across the split.
    assert " ".join(chunks).split() == words.split()


def test_whitespace_within_paragraph_is_normalized(tmp_path):
    doc = _write_doc(tmp_path, "This   paragraph\nhas irregular   whitespace and enough words.")
    chunks = EmbeddingBenchmark.chunk_document(doc, max_words=150, min_words=6)
    assert len(chunks) == 1
    assert "  " not in chunks[0]


def test_run_attaches_case_telemetry_after_measured_embedding(monkeypatch):
    class Engine:
        name = "fake"

        def ensure_running(self): return True
        def reachable_or_abort(self): return True
        def model_pulled(self, _tag): return True
        def embed(self, _tag, chunks): return EmbeddingMeasurement([[1.0]] * len(chunks), 0.5)
        def is_connection_crash(self, _exc): return False

    class Telemetry:
        def __init__(self): self.calls = []
        def begin_model_load(self): self.calls.append("load")
        def begin_measured(self, name): self.calls.append(name)
        def finish_case(self):
            self.calls.append("finish")
            return {"summary": {"process_rss_gb": {"peak_gb": 2}}}

    monkeypatch.setattr("scripts.workloads.embedding_benchmark.load_crash_cache", lambda _path: {})
    monkeypatch.setattr("scripts.workloads.embedding_benchmark.check_crash_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(EmbeddingBenchmark, "chunk_document", staticmethod(lambda: ["one chunk"]))
    monkeypatch.setattr("scripts.workloads.embedding_benchmark.config.N_RUNS", 1)
    telemetry = Telemetry()
    result = EmbeddingBenchmark().run(
        Engine(), [{"tag": "embed", "label": "Embed", "short": "embed"}],
        warmup_runs=1, telemetry=telemetry,
    )
    assert telemetry.calls == ["load", "measured:embedding", "finish"]
    assert result["embed"]["memory"]["summary"]["process_rss_gb"]["peak_gb"] == 2
