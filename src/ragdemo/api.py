"""FastAPI app. Uses Claude when ANTHROPIC_API_KEY is set, otherwise the offline extractive generator."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .generate import ClaudeGenerator, ExtractiveGenerator
from .pipeline import RagPipeline, build_pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


@lru_cache(maxsize=1)
def default_pipeline() -> RagPipeline:
    generator = ClaudeGenerator() if os.environ.get("ANTHROPIC_API_KEY") else ExtractiveGenerator()
    return build_pipeline(generator=generator)


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="RAG helpdesk demo", version="0.1.0")

    def get_pipeline() -> RagPipeline:
        return pipeline or default_pipeline()

    @app.get("/health")
    def health() -> dict:
        p = get_pipeline()
        return {"status": "ok", "chunks": len(p.index.chunks), "generator": type(p.generator).__name__}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        return get_pipeline().ask(request.question, k=request.k).to_dict()

    return app


app = create_app()
