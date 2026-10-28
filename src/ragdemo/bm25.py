"""A small, dependency-free BM25 index."""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunking import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or "
    "that the their there this to was what when where which who will with you your we our "
    "ask long many much take need get about any after am been before could did else even ever "
    "just may might more most not now only other out over please should so some than then these "
    "they those through too up us use very want would".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) < 2 or word in STOPWORDS:
            continue
        # Light stemming so "refunds"/"refunded"/"refund" meet in the middle.
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        tokens.append(word)
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf, length in zip(self.chunks, self._tf, self._len):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += self._idf[term] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
