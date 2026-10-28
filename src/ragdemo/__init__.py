"""Small retrieval-augmented generation demo: BM25 retrieval, cited answers, an eval harness."""

from .pipeline import Answer, RagPipeline, build_pipeline

__all__ = ["Answer", "RagPipeline", "build_pipeline"]
