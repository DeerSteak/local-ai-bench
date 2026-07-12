"""
embedding_benchmark.py — real document-ingestion workload: chunk one real
document the way a RAG pipeline would (paragraph-sized pieces), then embed
every chunk from it in a single call, the way an app actually ingests one
document — rather than sweeping arbitrary "batch sizes" that don't
correspond to any real client behavior.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

import config
from shared import Shared


class EmbeddingBenchmark:
    # max_words caps every chunk well under any embedding model's context
    # length (mxbai-embed-large's is 512 tokens) regardless of how the source
    # document is formatted — this is also what fixes "content length exceeds
    # context length" errors that an unbounded chunker can produce when it
    # runs into a document's own markdown tables/code blocks.
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

    @staticmethod
    def _load_crash_cache() -> dict:
        try:
            return json.loads(EmbeddingBenchmark.EMBED_CRASH_CACHE.read_text())
        except Exception:
            return {}

    @staticmethod
    def _save_crash_cache(cache: dict) -> None:
        try:
            EmbeddingBenchmark.EMBED_CRASH_CACHE.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            Shared.warn(f"Failed to save embedding crash cache: {e}")

    def run(self, models, warmup_runs=config.WARMUP_RUNS, save_fn=None):
        results = {}

        if not Shared.ollama_available():
            Shared.err("Ollama not running — skipping embedding benchmarks")
            return results

        crash_cache = EmbeddingBenchmark._load_crash_cache()
        chunks = EmbeddingBenchmark.chunk_document()
        Shared.log(f"Corpus: {len(chunks)} chunks from {EmbeddingBenchmark.EMBED_DOCUMENT_PATH.name} "
                   f"(max {EmbeddingBenchmark.EMBED_CHUNK_MAX_WORDS} words/chunk)")

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"Embeddings: {label}")

            try:
                if not Shared.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn(f"Pull with: ollama pull {tag}")
                    continue

                Shared.ok(f"Using Ollama model: {tag}")

                if tag in crash_cache:
                    detail = crash_cache[tag]
                    Shared.warn(f"{tag} previously crashed Ollama's runner repeatedly on "
                                f"{detail.get('crashed_at', 'an earlier run')} — skipping "
                                f"(delete {EmbeddingBenchmark.EMBED_CRASH_CACHE} to retry)")
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "known_crash",
                        "skip_detail": f"Crashed Ollama's runner repeatedly on {detail.get('crashed_at', 'an earlier run')}",
                    }
                    continue

                # Warm up model (load into memory) before measuring. The first embed
                # call against a freshly-unloaded model pays a one-time cost, model
                # weights loading into memory, first-call kernel/graph setup, that
                # has nothing to do with steady-state throughput — folding it into a
                # measured run would understate this model's real performance.
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
                        if isinstance(e, requests.exceptions.ConnectionError) or "actively refused" in str(e):
                            wait_t0 = time.perf_counter()
                            while time.perf_counter() - wait_t0 < 30:
                                if Shared.ollama_available():
                                    break
                                time.sleep(2)

                Shared.log(f"Embedding {len(chunks)} chunks in one call — {config.N_RUNS} runs ...")
                rates = []

                MAX_CRASH_RETRIES = 2
                run_i = 0
                crash_retries = 0
                gave_up_from_crashes = False
                while run_i < config.N_RUNS:
                    t0 = time.perf_counter()
                    try:
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
                        rates.append(rate)
                        print(f"    run {run_i+1}/{config.N_RUNS}: {rate:.0f} chunks/sec")
                        run_i += 1
                    except Exception as e:
                        Shared.err(f"Run {run_i+1} failed: {e}")
                        # A connection-refused error means Ollama's model runner
                        # subprocess had already died (commonly OOM) before this
                        # request. Wait for the main Ollama server to notice and
                        # respawn it, then retry this same run — up to a capped
                        # number of attempts, since a deterministic crash on this
                        # document would just recur identically forever.
                        if isinstance(e, requests.exceptions.ConnectionError) or "actively refused" in str(e):
                            crash_retries += 1
                            Shared.err(f"Ollama's model runner appears to have crashed embedding {tag} "
                                       f"— last server output:\n{Shared.tail_ollama_log()}")
                            if crash_retries > MAX_CRASH_RETRIES:
                                Shared.err(f"Ollama's model runner crashed {crash_retries} times — giving up on {tag}")
                                gave_up_from_crashes = True
                                break
                            Shared.warn(f"Waiting for recovery, retry {crash_retries}/{MAX_CRASH_RETRIES} ...")
                            recovered = False
                            wait_t0 = time.perf_counter()
                            while time.perf_counter() - wait_t0 < 30:
                                if Shared.ollama_available():
                                    recovered = True
                                    break
                                time.sleep(2)
                            if not recovered:
                                Shared.warn("Ollama did not become reachable again within 30s — giving up on this model")
                                break
                            # don't advance run_i — retry the same run now that Ollama is back
                        else:
                            run_i += 1

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
                elif gave_up_from_crashes:
                    crashed_at = datetime.now().isoformat(timespec="seconds")
                    crash_cache[tag] = {"crashed_at": crashed_at}
                    EmbeddingBenchmark._save_crash_cache(crash_cache)
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "known_crash",
                        "skip_detail": f"Ollama's runner crashed repeatedly embedding this document ({crashed_at})",
                    }
            finally:
                if save_fn:
                    save_fn(results)

        return results
