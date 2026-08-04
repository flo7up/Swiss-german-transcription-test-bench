"""Deterministic transcription metrics used by every benchmark run."""

from __future__ import annotations

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