"""Leakage-filtered SwissDial text pairs used for few-shot prompting and retrieval."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.casefold()) if len(token) > 1]


@dataclass(frozen=True)
class ExamplePair:
    sentence_id: int
    dialect_text: str
    standard_text: str
    tokens: frozenset[str]


class ExamplePool:
    """Per-dialect parallel sentences that never include an evaluated sentence ID."""

    def __init__(self, pairs_by_dialect: dict[str, list[ExamplePair]]) -> None:
        self._pairs = pairs_by_dialect
        self._idf: dict[str, dict[str, float]] = {}
        for dialect, pairs in pairs_by_dialect.items():
            document_frequency = Counter(token for pair in pairs for token in pair.tokens)
            total = len(pairs)
            self._idf[dialect] = {
                token: math.log(1 + total / count) for token, count in document_frequency.items()
            }

    @classmethod
    def load(cls, path: Path) -> "ExamplePool | None":
        if not path.is_file():
            return None
        pairs: dict[str, list[ExamplePair]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for dialect, text in record.get("dialects", {}).items():
                pairs.setdefault(dialect.upper(), []).append(
                    ExamplePair(
                        sentence_id=int(record["sentence_id"]),
                        dialect_text=text,
                        standard_text=record["de"],
                        tokens=frozenset(_tokens(text)),
                    )
                )
        return cls(pairs)

    def __contains__(self, dialect: str) -> bool:
        return bool(self._pairs.get(dialect.upper()))

    def sample(self, dialect: str, key: str, count: int, exclude_sentence_id: object = None) -> list[tuple[str, str]]:
        """Deterministic pseudo-random examples; the key keeps choices stable per evaluated clip."""
        candidates = [
            pair for pair in self._pairs.get(dialect.upper(), []) if str(pair.sentence_id) != str(exclude_sentence_id)
        ]
        candidates.sort(key=lambda pair: hashlib.sha1(f"{key}:{pair.sentence_id}".encode()).hexdigest())
        return [(pair.dialect_text, pair.standard_text) for pair in candidates[:count]]

    def similar(
        self, dialect: str, query: str, count: int, exclude_sentence_id: object = None
    ) -> list[tuple[str, str]]:
        """Return pairs sharing the most informative (IDF-weighted) dialect words with the query."""
        dialect = dialect.upper()
        idf = self._idf.get(dialect, {})
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []
        scored = []
        for pair in self._pairs.get(dialect, []):
            if str(pair.sentence_id) == str(exclude_sentence_id):
                continue
            overlap = query_tokens & pair.tokens
            if overlap:
                score = sum(idf.get(token, 0.0) for token in overlap) / math.sqrt(len(pair.tokens) + 1)
                scored.append((score, pair.sentence_id, pair))
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [(pair.dialect_text, pair.standard_text) for _, _, pair in scored[:count]]
