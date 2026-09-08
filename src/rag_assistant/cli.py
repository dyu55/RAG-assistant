from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .config import Settings
from .demo import DOCUMENTS, QUESTIONS
from .models import Question
from .service import KnowledgeService


def evaluate() -> dict:
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as directory:
        service = KnowledgeService(Settings(data_dir=Path(directory)))
        for name, text in DOCUMENTS.items():
            service.ingest(name, text.encode())
        cases = []
        for question, expected in QUESTIONS:
            answer = service.ask(Question(text=question))
            cases.append(
                {
                    "question": question,
                    "expected_source": expected,
                    "source_recalled": any(e.filename == expected for e in answer.evidence),
                    "supported": answer.status == "supported",
                    "latency_ms": answer.timings_ms["total"],
                }
            )
        unknown = service.ask(Question(text="What is the orbital period of Neptune?"))
        return {
            "dataset": "bundled fictional Atlas corpus",
            "cases": cases,
            "recall_at_5": sum(c["source_recalled"] for c in cases) / len(cases),
            "unknown_question_abstained": unknown.status == "abstained",
            "mode": "offline lexical vectors and extractive answers",
            "scope": "Small deterministic smoke evaluation, not a general model-quality benchmark.",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG Assistant — local evidence workbench")
    parser.add_argument("--data-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Open the local web application")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("demo", help="Import the fictional Atlas example documents")
    ingest = commands.add_parser("ingest", help="Import UTF-8 documents or PDFs")
    ingest.add_argument("files", type=Path, nargs="+")
    ask = commands.add_parser("ask", help="Ask a question and return JSON with evidence")
    ask.add_argument("question")
    commands.add_parser("evaluate", help="Run the bundled offline retrieval evaluation")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
            return 0
        settings = Settings.from_env()
        if args.data_dir:
            settings.data_dir = args.data_dir
        if args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port)
            return 0
        service = KnowledgeService(settings)
        if args.command == "ask":
            print(service.ask(Question(text=args.question)).model_dump_json(indent=2))
        else:
            items = (
                DOCUMENTS.items()
                if args.command == "demo"
                else [(p.name, p.read_bytes()) for p in args.files]
            )
            for name, content in items:
                print(
                    json.dumps(
                        service.ingest(
                            name, content.encode() if isinstance(content, str) else content
                        )
                    )
                )
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
