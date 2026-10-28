"""Split markdown documents into retrievable chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str        # "<doc_id>#<n>", unique and stable for a given corpus
    doc_id: str    # file stem, e.g. "refunds"
    heading: str   # nearest markdown heading, "" if none
    text: str      # heading + body; this is what gets indexed and shown to the model


def split_markdown(doc_id: str, markdown: str, max_words: int = 120) -> list[Chunk]:
    """Chunk by heading, then by paragraph if a section is longer than ``max_words``."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            sections.append((heading.group(1).strip(), []))
        else:
            sections[-1][1].append(line)

    chunks: list[Chunk] = []
    for heading, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        for body in _pack(paragraphs, max_words):
            text = f"{heading}\n{body}" if heading else body
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, heading, text))
    return chunks


def _pack(paragraphs: list[str], max_words: int) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in paragraphs:
        words = len(paragraph.split())
        if current and count + words > max_words:
            packed.append("\n\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        packed.append("\n\n".join(current))
    return packed


def load_corpus(directory: str | Path, max_words: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        chunks.extend(split_markdown(path.stem, path.read_text(encoding="utf-8"), max_words))
    if not chunks:
        raise FileNotFoundError(f"no markdown documents found in {directory}")
    return chunks
