from embedding_benchmark import EmbeddingBenchmark


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
