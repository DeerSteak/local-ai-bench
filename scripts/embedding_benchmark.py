"""
embedding_benchmark.py — real document-ingestion workload: chunk one document
into paragraph-sized pieces (as a RAG pipeline would), then embed every chunk
in a single call, the way an app ingests one document — rather than sweeping
arbitrary batch sizes that match no real client behavior.
"""

import re
import time
from pathlib import Path

import requests

import config
from shared import Shared


class EmbeddingBenchmark:
    # max_words caps every chunk well under any embedding model's context
    # length (mxbai-embed-large's is 512 tokens), which also avoids "content
    # length exceeds context length" errors an unbounded chunker hits on a
    # document's own markdown tables/code blocks.
    EMBED_DOCUMENT_PATH = config.SCRIPT_DIR / "sample_document.txt"
    EMBED_CHUNK_MAX_WORDS = 150
    EMBED_CHUNK_MIN_WORDS = 6

    # Records model/document combos that crashed Ollama's runner repeatedly
    # (deterministically, not a transient blip) so future runs don't waste time
    # rediscovering the same crash. Delete this file to retry a skipped model.
    EMBED_CRASH_CACHE = Path(".embed_crash_cache.json")

    @staticmethod
    def chunk_document(path: Path = EMBED_DOCUMENT_PATH,
                        max_words: int = EMBED_CHUNK_MAX_WORDS,
                        min_words: int = EMBED_CHUNK_MIN_WORDS) -> list[str]:
        """Split a document into paragraph-sized chunks, each capped at
        max_words. Paragraphs longer than max_words are packed sentence-by-
        sentence up to the cap; anything that's still too long after that
        (e.g. a code block or table with no sentence punctuation) is hard-split
        by word count so no chunk can ever exceed the cap."""
        paragraphs = [p.strip() for p in path.read_text().split("\n\n") if p.strip()]

        def split_oversized(words: list[str]) -> list[str]:
            return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

        chunks = []
        for para in paragraphs:
            words = " ".join(para.split()).split()
            if len(words) < min_words:
                continue
            if len(words) <= max_words:
                chunks.append(" ".join(words))
                continue

            current, current_len = [], 0
            for sentence in re.split(r"(?<=[.!?])\s+", " ".join(words)):
                sentence_words = sentence.split()
                if len(sentence_words) > max_words:
                    if current:
                        chunks.append(" ".join(current))
                        current, current_len = [], 0
                    chunks.extend(split_oversized(sentence_words))
                    continue
                if current_len + len(sentence_words) > max_words and current:
                    chunks.append(" ".join(current))
                    current, current_len = [], 0
                current.extend(sentence_words)
                current_len += len(sentence_words)
            if current:
                chunks.append(" ".join(current))

        return chunks

    def run(self, models, warmup_runs=config.WARMUP_RUNS, save_fn=None):  # pragma: no cover — orchestrates real Ollama runs
        results = {}

        if not Shared.ollama_available():
            Shared.err("Ollama not running — skipping embedding benchmarks")
            return results

        crash_cache = Shared.load_crash_cache(EmbeddingBenchmark.EMBED_CRASH_CACHE)
        chunks = EmbeddingBenchmark.chunk_document()
        Shared.log(f"Corpus: {len(chunks)} chunks from {EmbeddingBenchmark.EMBED_DOCUMENT_PATH.name} "
                   f"(max {EmbeddingBenchmark.EMBED_CHUNK_MAX_WORDS} words/chunk)")

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"Embeddings: {label}")

            if not Shared.ollama_reachable_or_abort():
                break

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                Shared.ok(f"Using Ollama model: {tag}")

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, EmbeddingBenchmark.EMBED_CRASH_CACHE)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                # Warm up before measuring: the first embed call against a
                # freshly-unloaded model pays a one-time load/setup cost unrelated
                # to steady-state throughput, and folding it into a measured run
                # would understate performance.
                Shared.log(f"Warming up {label} ...")
                for warmup_i in range(warmup_runs):
                    try:
                        resp = requests.post(
                            f"{config.OLLAMA_URL}/api/embed",
                            json={"model": tag, "input": chunks},
                            timeout=120,
                        )
                        if not resp.ok:
                            Shared.warn(f"Warmup run {warmup_i+1} failed: HTTP {resp.status_code}")
                        else:
                            Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
                    except Exception as e:
                        Shared.warn(f"Warmup run {warmup_i+1} failed: {e}")
                        if Shared.is_connection_crash(e):
                            Shared.wait_for_ollama_recovery()

                Shared.log(f"Embedding {len(chunks)} chunks in one call — {config.N_RUNS} runs ...")

                def _embed_once(run_i):
                    t0 = time.perf_counter()
                    resp = requests.post(
                        f"{config.OLLAMA_URL}/api/embed",
                        json={"model": tag, "input": chunks},
                        timeout=120,
                    )
                    if not resp.ok:
                        try:
                            detail = resp.json()
                        except Exception:
                            detail = resp.text[:500]
                        raise RuntimeError(
                            f"Ollama rejected embed request (HTTP {resp.status_code}, "
                            f"n_chunks={len(chunks)}): {detail}"
                        )
                    elapsed = time.perf_counter() - t0
                    rate = len(chunks) / elapsed
                    print(f"    run {run_i+1}/{config.N_RUNS}: {rate:.0f} chunks/sec")
                    return rate

                rates, status = Shared.run_measured_calls(
                    config.N_RUNS, _embed_once, tag, crash_cache,
                    EmbeddingBenchmark.EMBED_CRASH_CACHE, "embedding this document")

                if rates:
                    results[short] = {
                        "label": label,
                        "chunks_per_sec_mean":  round(Shared.mean(rates), 1),
                        "chunks_per_sec_stdev": round(Shared.stdev(rates), 1),
                        "device":               "gpu",
                        "n_chunks":             len(chunks),
                        "n_runs":               len(rates),
                        "runs":                [round(r, 1) for r in rates],
                    }
                    Shared.ok(f"{label}: {results[short]['chunks_per_sec_mean']:.0f} chunks/sec")
                elif status == "crashed":
                    crashed_at = crash_cache.get(tag, {}).get("crashed_at", "an earlier run")
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "known_crash",
                        "skip_detail": f"Ollama's runner crashed repeatedly embedding this document ({crashed_at})",
                    }
                elif status == "timed_out":
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "timed_out",
                        "skip_detail": "Embedding run timed out (120s)",
                    }
                else:
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "failed",
                        "skip_detail": "All embedding runs failed",
                    }
            finally:
                if save_fn:
                    save_fn(results)

        return results
