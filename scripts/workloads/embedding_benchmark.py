"""Real document-ingestion workload — see docs/workloads.md#embeddings."""

import re
import time
from pathlib import Path

from scripts.runtime import config
from scripts.runtime.engines.base import embedding_validation_errors
from scripts.runtime.shared import Shared
from scripts.app.progress_events import emit_model_finished, emit_progress


class EmbeddingBenchmark:
    EMBED_DOCUMENT_PATH = Path(__file__).with_name("data") / "sample_document.txt"
    EMBED_CHUNK_MAX_WORDS = 150
    EMBED_CHUNK_MIN_WORDS = 6

    EMBED_CRASH_CACHE = Path(".embed_crash_cache.json")  # see docs/project-structure.md

    @staticmethod
    def chunk_document(path: Path = EMBED_DOCUMENT_PATH,
                        max_words: int = EMBED_CHUNK_MAX_WORDS,
                        min_words: int = EMBED_CHUNK_MIN_WORDS) -> list[str]:
        """Split into paragraph-sized chunks capped at max_words, packing
        oversized paragraphs sentence-by-sentence and hard-splitting the rest."""
        paragraphs = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]

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

    def run(self, engine, models, warmup_runs=config.WARMUP_RUNS, save_fn=None):  # pragma: no cover — orchestrates real engine runs
        results = {}

        if not engine.ensure_running():
            Shared.err("Inference engine not running — skipping embedding benchmarks")
            return results

        crash_cache = Shared.load_crash_cache(EmbeddingBenchmark.EMBED_CRASH_CACHE)
        chunks = EmbeddingBenchmark.chunk_document()
        Shared.log(f"Corpus: {len(chunks)} chunks from {EmbeddingBenchmark.EMBED_DOCUMENT_PATH.name} "
                   f"(max {EmbeddingBenchmark.EMBED_CHUNK_MAX_WORDS} words/chunk)")

        for model in models:
            tag   = model["tag"]
            label = model["label"]
            short = model["short"]

            Shared.section(f"Embeddings ({engine.name}): {label}")

            if not engine.reachable_or_abort():
                break

            emit_progress("model", "emb", "running", label)
            try:
                if not engine.model_pulled(tag):
                    Shared.warn(f"{tag} not pulled — skipping")
                    Shared.warn("Download it with: python setup_check.py")
                    continue

                Shared.ok(f"Using model: {tag}")

                skip_entry = Shared.check_crash_cache(tag, label, crash_cache, EmbeddingBenchmark.EMBED_CRASH_CACHE)
                if skip_entry is not None:
                    results[short] = skip_entry
                    continue

                Shared.log(f"Warming up {label} ...")
                for warmup_i in range(warmup_runs):
                    try:
                        engine.embed(tag, chunks)
                        Shared.log(f"Warmup run {warmup_i+1}/{warmup_runs} done")
                    except Exception as e:
                        Shared.warn(f"Warmup run {warmup_i+1} failed: {e}")
                        if engine.is_connection_crash(e):
                            engine.wait_for_recovery()

                Shared.log(f"Embedding {len(chunks)} chunks in one call — {config.N_RUNS} runs ...")

                def _embed_once(run_i):
                    measurement = engine.embed(tag, chunks)
                    if not embedding_validation_errors(measurement):
                        rate = len(chunks) / measurement.client_wall_sec
                        Shared.output(f"    run {run_i+1}/{config.N_RUNS}: {rate:.0f} chunks/sec")
                    return measurement

                samples, status, _, _metadata = Shared.run_measured_calls(
                    config.N_RUNS, _embed_once, tag, crash_cache,
                    EmbeddingBenchmark.EMBED_CRASH_CACHE, "embedding this document", engine)

                valid_samples = [sample for sample in samples
                                 if not embedding_validation_errors(sample)]
                rates = [len(chunks) / sample.client_wall_sec for sample in valid_samples]
                if rates:
                    results[short] = {
                        "label": label,
                        "chunks_per_sec_mean":  round(Shared.mean(rates), 1),
                        "chunks_per_sec_stdev": round(Shared.stdev(rates), 1),
                        "device":               "gpu",
                        "n_chunks":             len(chunks),
                        "n_runs":               len(samples),
                        "requested_runs":       config.N_RUNS,
                        "completed_runs":       len(samples),
                        "valid_runs":           len(valid_samples),
                        "invalid_runs": [
                            {"run": index + 1, "errors": embedding_validation_errors(sample)}
                            for index, sample in enumerate(samples)
                            if embedding_validation_errors(sample)
                        ],
                        "client_wall_runs_sec": [
                            round(sample.client_wall_sec, 3) for sample in valid_samples
                        ],
                        "runs":                [round(r, 1) for r in rates],
                    }
                    if len(valid_samples) >= 2:
                        results[short].update({
                            "chunks_per_sec_median": round(Shared.median(rates), 1),
                            "chunks_per_sec_cv": round(Shared.coefficient_of_variation(rates), 4),
                        })
                    Shared.ok(f"{label}: {results[short]['chunks_per_sec_mean']:.0f} chunks/sec")
                elif status == "crashed":
                    crashed_at = crash_cache.get(tag, {}).get("crashed_at", "an earlier run")
                    results[short] = {
                        "label": label,
                        "skipped": True,
                        "skip_reason": "known_crash",
                        "skip_detail": f"The engine's runner crashed repeatedly embedding this document ({crashed_at})",
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
                emit_model_finished("emb", label, results.get(short))

        return results
