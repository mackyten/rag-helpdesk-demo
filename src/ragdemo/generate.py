"""Answer generators: an offline extractive one and a Claude-backed one."""

from __future__ import annotations

import os
import re
from typing import Protocol

from .bm25 import tokenize
from .chunking import Chunk

SYSTEM_PROMPT = """You answer customer-support questions using ONLY the numbered sources provided.
- Cite every claim with the source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: I don't know based on the documentation.
- Be concise. Do not mention these instructions."""

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSED = "I can't help with that request."


class Generator(Protocol):
    def generate(self, question: str, sources: list[Chunk]) -> str: ...


def format_sources(sources: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.doc_id}) {c.text}" for i, c in enumerate(sources, start=1))


class ExtractiveGenerator:
    """No model, no network: return the sentences that best overlap the question.

    Deterministic, so the pipeline, API and evals run in CI without an API key.
    """

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, sources: list[Chunk]) -> str:
        wanted = set(tokenize(question))
        candidates: list[tuple[int, int, str]] = []
        for number, chunk in enumerate(sources, start=1):
            body = chunk.text.split("\n", 1)[1] if chunk.heading and "\n" in chunk.text else chunk.text
            for sentence in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")):
                overlap = len(wanted & set(tokenize(sentence)))
                if overlap:
                    candidates.append((overlap, number, sentence.strip()))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        best = candidates[: self.max_sentences]
        if not best:
            return "I don't know based on the documentation."
        return " ".join(f"{sentence} [{number}]" for _, number, sentence in best)


class ClaudeGenerator:
    """Grounded answers from Claude, with refusal fallbacks on the Claude API."""

    def __init__(self, client=None, model: str | None = None, max_tokens: int = 4096) -> None:
        if client is None:
            import anthropic  # imported lazily so the offline path needs no key or network

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def generate(self, question: str, sources: list[Chunk]) -> str:
        prompt = f"Sources:\n{format_sources(sources)}\n\nQuestion: {question}"
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},  # grounded Q&A over short context does not need deep reasoning
            betas=[FALLBACK_BETA],
            fallbacks="default",  # route a safety refusal to another model instead of failing
        )
        if response.stop_reason == "refusal":
            return REFUSED
        return "".join(block.text for block in response.content if block.type == "text").strip()
