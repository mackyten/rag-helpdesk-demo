# RAG Helpdesk Demo

[![CI](https://github.com/mackyten/rag-helpdesk-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/mackyten/rag-helpdesk-demo/actions/workflows/ci.yml)

A small, readable retrieval-augmented generation (RAG) service for a support knowledge base. It retrieves the relevant passages, asks Claude to answer **only from them with numbered citations**, refuses when the documentation doesn't contain the answer, and ships with an **eval harness** so changes can be measured instead of guessed.

The knowledge base is six short documents for a fictional product ("Brightbox"), written for this demo. Nothing here comes from a real company or customer.

## What it shows

- **Retrieval:** a dependency-free BM25 index over heading-aware chunks (`chunking.py`, `bm25.py`)
- **Grounded generation:** a `ClaudeGenerator` that prompts with numbered sources, requires `[n]` citations and an exact "I don't know" fallback, and handles safety refusals with server-side model fallbacks (`generate.py`)
- **Abstention:** if the best passage scores below a threshold, the pipeline answers "I don't know" without calling the model at all, which saves a request and avoids a confident wrong answer (`pipeline.py`)
- **Citation checking:** every `[n]` in the answer is verified against what was actually retrieved; invented numbers are reported as `ungrounded_citations`
- **Evals:** retrieval hit@1 / hit@k / MRR, answer correctness, groundedness and abstention on a golden set, runnable offline and usable as a CI gate (`evals.py`)
- **HTTP API and CLI:** FastAPI (`/ask`, `/health`) and `python -m ragdemo ask|eval|serve`
- **Offline mode:** an `ExtractiveGenerator` needs no model or network, so tests and CI run without an API key

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m ragdemo --generator extractive ask "How long is the hardware warranty?"
python -m ragdemo --generator extractive eval
pytest
```

With Claude:

```bash
export ANTHROPIC_API_KEY=...        # optional: ANTHROPIC_MODEL (default claude-opus-5-5)
python -m ragdemo --generator claude ask "Does the warranty cover liquid damage?"
python -m ragdemo serve             # http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What formats can I export my data in?"}'
```

```json
{
  "question": "What formats can I export my data in?",
  "answer": "Open Settings, choose \"Export data\" and pick a format: JSON or CSV. [1]",
  "abstained": false,
  "citations": [{"number": 1, "chunk_id": "data-privacy#1", "doc_id": "data-privacy", "heading": "Exporting your data", "score": 10.5}],
  "retrieved": ["..."],
  "ungrounded_citations": []
}
```

## Eval results

The golden set (`evals/golden.jsonl`) has 18 answerable questions and 6 the corpus cannot answer, three of them deliberately close to real topics ("delivery time for a pizza"). Baseline with the offline extractive generator:

| Metric | Score |
|---|---|
| Retrieval hit@1 | 94.4% |
| Retrieval hit@4 | 100% |
| MRR | 97.2% |
| Answer correctness (required phrases present) | 88.9% |
| Groundedness (cites only real sources) | 100% |
| Abstains on unanswerable questions | 100% |

The report lists its own failures. For example, "How long until a refund reaches my card?" retrieves the right passage but the extractive generator quotes the wrong sentence, which is exactly the gap an LLM generator is meant to close. Run `python -m ragdemo eval` to see the current list. CI fails if retrieval hit@4 or abstention drops below 100%.

## How abstention is tuned

Answerable questions score 4.25 or higher at their best passage; unanswerable ones score 2.8 or lower (or match nothing). The default `min_score` is 3.5, set between the two. That is tuned on a small set, so treat it as a starting point and re-tune when the corpus or the golden set changes.

## Design notes and limitations

- **BM25 is a deliberate first stage.** It is fast, explainable and needs no embedding service. It cannot tell that "reset the password for my email provider" is about a different product than the Brightbox reset-password docs (it scores 8.5), so on a real corpus I would add a dense or hybrid retriever behind the same `index.search` interface and a reranker.
- **The correctness grader is a substring check.** It is cheap and deterministic, but it can pass for the wrong reason (question q12 passes on a different passage that happens to contain the same number). For free-text answers I would add an LLM judge, checked against human labels.
- **The Claude path is tested without a live call.** Unit tests assert the exact request (model, system prompt, `effort`, fallback beta) against a fake client, and the request body was also checked against the real SDK through a local stub server. Running against the live API needs your own key.
- **Scale.** The index is in memory and rebuilt at startup, which is right for thousands of chunks and wrong for millions.
- The corpus is tiny and synthetic, so the metrics demonstrate the harness, not production accuracy.

## Layout

```
data/docs/            sample knowledge base (markdown)
evals/golden.jsonl    questions, expected document, required phrases
src/ragdemo/
  chunking.py         heading-aware chunking
  bm25.py             BM25 index
  generate.py         ExtractiveGenerator, ClaudeGenerator
  pipeline.py         retrieve, abstain, generate, check citations
  evals.py            metrics and report
  api.py, __main__.py FastAPI app and CLI
tests/                16 tests
```

## License

[MIT](LICENSE)
