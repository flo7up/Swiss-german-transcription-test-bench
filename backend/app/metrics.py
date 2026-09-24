"""Deterministic transcription metrics used by every benchmark run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class TranscriptScores:
    """Objective scores for a hypothesis against a reference transcript."""

    word_error_rate: float | None
    character_error_rate: float | None
    reference_word_count: int
    hypothesis_word_count: int


def word_match_rate(word_error_rate: float | None) -> float | None:
    """Return an intuitive bounded match percentage derived from WER."""
    if word_error_rate is None:
        return None
    return max(0, 1 - word_error_rate)


def normalize_transcript(text: str) -> str:
    """Normalize formatting while retaining Swiss German letters and word boundaries."""
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def chrf_score(reference: str | None, hypothesis: str | None, max_order: int = 6, beta: float = 2.0) -> float | None:
    """Return chrF (character n-gram F-score, 0-1) on normalized text; tolerant of inflection and paraphrase."""
    if reference is None or hypothesis is None:
        return None
    reference_chars = normalize_transcript(reference).replace(" ", "")
    hypothesis_chars = normalize_transcript(hypothesis).replace(" ", "")
    if not reference_chars:
        return None
    if not hypothesis_chars:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, max_order + 1):
        reference_ngrams = Counter(reference_chars[i : i + order] for i in range(len(reference_chars) - order + 1))
        hypothesis_ngrams = Counter(hypothesis_chars[i : i + order] for i in range(len(hypothesis_chars) - order + 1))
        if not reference_ngrams or not hypothesis_ngrams:
            break
        matches = sum((reference_ngrams & hypothesis_ngrams).values())
        precisions.append(matches / sum(hypothesis_ngrams.values()))
        recalls.append(matches / sum(reference_ngrams.values()))

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision == 0 and recall == 0:
        return 0.0
    beta_squared = beta**2
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Return the Levenshtein edit distance using one rolling row of memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_value in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_value in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_value == hypothesis_value else 1
            current.append(
                min(
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def score_transcript(reference: str | None, hypothesis: str) -> TranscriptScores:
    """Score a model transcript, returning unavailable rates when no reference exists."""
    reference_normalized = normalize_transcript(reference or "")
    hypothesis_normalized = normalize_transcript(hypothesis)
    reference_words = reference_normalized.split()
    hypothesis_words = hypothesis_normalized.split()

    if not reference_words:
        return TranscriptScores(
            word_error_rate=None,
            character_error_rate=None,
            reference_word_count=0,
            hypothesis_word_count=len(hypothesis_words),
        )

    word_error_rate = _edit_distance(reference_words, hypothesis_words) / len(reference_words)
    reference_characters = list(reference_normalized)
    hypothesis_characters = list(hypothesis_normalized)
    character_error_rate = _edit_distance(reference_characters, hypothesis_characters) / len(reference_characters)

    return TranscriptScores(
        word_error_rate=word_error_rate,
        character_error_rate=character_error_rate,
        reference_word_count=len(reference_words),
        hypothesis_word_count=len(hypothesis_words),
    )