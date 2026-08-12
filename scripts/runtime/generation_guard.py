"""Heuristics for stopping degenerate streamed generation loops."""

LOOP_HEDGE_PHRASES_HIGH_THRESHOLD = ["wait,", "wait -", "actually,", "hold on,"]
LOOP_HEDGE_PHRASES = [
    "let me reconsider", "let me recalculate", "let me re-check", "let me recheck",
    "let me recompute", "let me redo", "let me try again", "let's try again",
    "let's recalculate", "on second thought", "there seems to be a mistake",
    "there seems to have been", "i made an error", "i made a mistake", "that's not right",
    "this is incorrect", "let's start over", "apolog", "correcting myself",
    "let me reevaluate", "let me re-evaluate",
]


def has_repeated_verbatim_ngram(text: str, ngram_words: int = 12,
                                min_repeats: int = 3) -> bool:
    words = text.split()
    if len(words) < ngram_words * min_repeats:
        return False
    seen: dict[str, int] = {}
    for index in range(len(words) - ngram_words + 1):
        gram = " ".join(words[index:index + ngram_words])
        count = seen.get(gram, 0) + 1
        if count >= min_repeats:
            return True
        seen[gram] = count
    return False


def has_repeated_hedging_phrase(text: str, min_repeats: int = 3,
                                high_threshold_repeats: int = 5) -> bool:
    lowered = text.lower()
    return (
        any(lowered.count(phrase) >= min_repeats for phrase in LOOP_HEDGE_PHRASES)
        or any(lowered.count(phrase) >= high_threshold_repeats
               for phrase in LOOP_HEDGE_PHRASES_HIGH_THRESHOLD)
    )


def looks_like_loop(text: str, ngram_words: int = 12, min_repeats: int = 3,
                    hedge_min_repeats: int = 3,
                    hedge_high_threshold_repeats: int = 5) -> bool:
    return (
        has_repeated_verbatim_ngram(text, ngram_words, min_repeats)
        or has_repeated_hedging_phrase(text, hedge_min_repeats, hedge_high_threshold_repeats)
    )
