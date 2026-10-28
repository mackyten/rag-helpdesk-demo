"""Offline eval harness: retrieval quality, answer correctness, grounding and abstention.

Golden rows are JSON lines: ``{"id", "question", "doc", "must_include"}``. ``doc`` is the file
stem that holds the answer, or ``null`` for a question the corpus cannot answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .pipeline import RagPipeline

DEFAULT_GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden.jsonl"


@dataclass
class CaseResult:
    id: str
    answerable: bool
    rank: int | None              # 1-based rank of the expected document, None if not retrieved
    abstained: bool
    correct: bool | None          # all must_include phrases present; None for unanswerable rows
    grounded: bool | None         # cites at least one real source and none that do not exist
    answer: str = ""


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)
    k: int = 4

    def _answerable(self) -> list[CaseResult]:
        return [c for c in self.cases if c.answerable]

    def _unanswerable(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.answerable]

    @staticmethod
    def _rate(hits: int, total: int) -> float:
        return hits / total if total else 0.0

    @property
    def hit_at_1(self) -> float:
        rows = self._answerable()
        return self._rate(sum(1 for c in rows if c.rank == 1), len(rows))

    @property
    def hit_at_k(self) -> float:
        rows = self._answerable()
        return self._rate(sum(1 for c in rows if c.rank is not None), len(rows))

    @property
    def mrr(self) -> float:
        rows = self._answerable()
        return sum(1 / c.rank for c in rows if c.rank) / len(rows) if rows else 0.0

    @property
    def answer_correctness(self) -> float:
        rows = self._answerable()
        return self._rate(sum(1 for c in rows if c.correct), len(rows))

    @property
    def groundedness(self) -> float:
        rows = [c for c in self._answerable() if not c.abstained]
        return self._rate(sum(1 for c in rows if c.grounded), len(rows))

    @property
    def abstain_rate(self) -> float:
        rows = self._unanswerable()
        return self._rate(sum(1 for c in rows if c.abstained), len(rows))

    def metrics(self) -> dict[str, float]:
        return {
            "hit_at_1": self.hit_at_1,
            f"hit_at_{self.k}": self.hit_at_k,
            "mrr": self.mrr,
            "answer_correctness": self.answer_correctness,
            "groundedness": self.groundedness,
            "abstain_on_unanswerable": self.abstain_rate,
        }

    def format(self) -> str:
        lines = [f"{name:<26}{value:>7.1%}" for name, value in self.metrics().items()]
        failures = [c for c in self.cases if self._failed(c)]
        if failures:
            lines.append("")
            lines.append("failures:")
            lines.extend(f"  {c.id}: rank={c.rank} abstained={c.abstained} correct={c.correct} answer={c.answer[:70]!r}" for c in failures)
        return "\n".join(lines)

    @staticmethod
    def _failed(c: CaseResult) -> bool:
        if not c.answerable:
            return not c.abstained
        return c.rank != 1 or not c.correct or c.grounded is False


def load_golden(path: str | Path = DEFAULT_GOLDEN) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_evals(pipeline: RagPipeline, golden: list[dict], k: int = 4) -> EvalReport:
    report = EvalReport(k=k)
    for row in golden:
        answer = pipeline.ask(row["question"], k=k)
        expected = row["doc"]
        docs = [c.doc_id for c in answer.retrieved]
        rank = docs.index(expected) + 1 if expected in docs else None
        if expected is None:
            report.cases.append(CaseResult(row["id"], False, None, answer.abstained, None, None, answer.answer))
            continue
        lowered = answer.answer.lower()
        correct = (not answer.abstained) and all(p.lower() in lowered for p in row["must_include"])
        grounded = bool(answer.citations) and not answer.ungrounded_citations
        report.cases.append(CaseResult(row["id"], True, rank, answer.abstained, correct, grounded, answer.answer))
    return report
