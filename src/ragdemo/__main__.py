"""Command line: ``python -m ragdemo ask "..."`` / ``eval`` / ``serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .evals import DEFAULT_GOLDEN, load_golden, run_evals
from .generate import ClaudeGenerator, ExtractiveGenerator
from .pipeline import build_pipeline


def _generator(name: str):
    if name == "claude" or (name == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        return ClaudeGenerator()
    return ExtractiveGenerator()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragdemo")
    parser.add_argument("--generator", choices=["auto", "extractive", "claude"], default="auto")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=4)

    evaluate = sub.add_parser("eval", help="score the golden set")
    evaluate.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    evaluate.add_argument("-k", type=int, default=4)
    evaluate.add_argument("--min-hit-at-k", type=float, default=0.0, help="exit 1 if retrieval hit@k is lower")
    evaluate.add_argument("--min-abstain", type=float, default=0.0, help="exit 1 if unanswerable abstention is lower")
    evaluate.add_argument("--json", action="store_true", help="print metrics as JSON")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("ragdemo.api:app", host="127.0.0.1", port=args.port)
        return 0

    pipeline = build_pipeline(generator=_generator(args.generator))
    if args.command == "ask":
        print(json.dumps(pipeline.ask(args.question, k=args.k).to_dict(), indent=2))
        return 0

    report = run_evals(pipeline, load_golden(args.golden), k=args.k)
    print(json.dumps(report.metrics(), indent=2) if args.json else report.format())
    ok = report.hit_at_k >= args.min_hit_at_k and report.abstain_rate >= args.min_abstain
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
