"""Retrieve, decide whether to answer, generate, and check the citations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .bm25 import BM25Index
from .chunking import Chunk, load_corpus
from .generate import ExtractiveGenerator, Generator

DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "data" / "docs"
ABSTAIN = "I don't know based on the documentation."


@dataclass
class Citation:
    number: int
    chunk_id: str
    doc_id: str
    heading: str
    score: float


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)  # sources the answer actually cites
    retrieved: list[Citation] = field(default_factory=list)  # everything that was retrieved
    ungrounded_citations: list[int] = field(default_factory=list)  # [n] markers with no such source

    def to_dict(self) -> dict:
        return asdict(self)


class RagPipeline:
    def __init__(self, index: BM25Index, generator: Generator, min_score: float = 3.5, top_k: int = 4) -> None:
        self.index = index
        self.generator = generator
        self.min_score = min_score
        self.top_k = top_k

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        return self.index.search(question, k or self.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k)
        retrieved = [Citation(i, c.id, c.doc_id, c.heading, round(s, 3)) for i, (c, s) in enumerate(hits, start=1)]

        # Abstain without calling the model when nothing in the corpus matches well enough.
        if not hits or hits[0][1] < self.min_score:
            return Answer(question, ABSTAIN, abstained=True, retrieved=retrieved)

        sources = [chunk for chunk, _ in hits]
        text = self.generator.generate(question, sources)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
        by_number = {c.number: c for c in retrieved}
        return Answer(
            question,
            text,
            abstained=text.strip().lower().startswith("i don't know"),
            citations=[by_number[n] for n in cited if n in by_number],
            retrieved=retrieved,
            ungrounded_citations=[n for n in cited if n not in by_number],
        )


def build_pipeline(
    docs_dir: str | Path = DEFAULT_DOCS,
    generator: Generator | None = None,
    min_score: float = 3.5,
    top_k: int = 4,
) -> RagPipeline:
    return RagPipeline(BM25Index(load_corpus(docs_dir)), generator or ExtractiveGenerator(), min_score, top_k)
