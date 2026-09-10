"""
Unit tests for RAG Triad and Groundedness Evaluation Engine.
"""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.evaluation import (
    compute_answer_relevance,
    compute_context_relevance,
    compute_groundedness,
    evaluate_triad,
)
from rag_assistant.models import Claim, Evidence, Question
from rag_assistant.service import KnowledgeService


def _make_evidence(source_id: str, text: str, relevance: float = 0.8) -> Evidence:
    return Evidence(
        source_id=source_id,
        chunk_id=f"chk_{source_id}",
        document_id="doc1",
        filename="doc1.txt",
        text=text,
        page=1,
        start=0,
        end=len(text),
        relevance=relevance,
        fusion_score=0.015,
        channels=["hybrid"],
    )


def test_context_relevance_empty():
    assert compute_context_relevance("", []) == 0.0
    assert compute_context_relevance("test query", []) == 0.0


def test_context_relevance_overlap():
    ev = [_make_evidence("S1", "FastAPI web framework with async routing.", 0.9)]
    score = compute_context_relevance("FastAPI async routing", ev)
    assert 0.7 <= score <= 1.0


def test_groundedness_empty_claims():
    assert compute_groundedness([], []) == 0.0


def test_groundedness_valid_claims():
    source_text = "FastAPI handles asynchronous web routing with high throughput."
    ev = [_make_evidence("S1", source_text)]
    claim = Claim(
        source_id="S1",
        text="FastAPI handles asynchronous web routing.",
        quote="FastAPI handles asynchronous web routing with high throughput.",
    )
    score = compute_groundedness([claim], ev)
    assert score > 0.8


def test_groundedness_quote_not_in_source():
    ev = [_make_evidence("S1", "Source text without the required facts.")]
    claim = Claim(
        source_id="S1",
        text="Hallucinated claim text.",
        quote="Fabricated quotation never present.",
    )
    assert compute_groundedness([claim], ev) == 0.0


def test_groundedness_missing_source():
    ev = [_make_evidence("S1", "Some text.")]
    claim = Claim(
        source_id="S99",
        text="Claim text.",
        quote="Some quote.",
    )
    assert compute_groundedness([claim], ev) == 0.0


def test_answer_relevance():
    query = "How does caching work?"
    answer = "Caching stores previous results in an in-memory map to reduce latency."
    score = compute_answer_relevance(query, answer)
    assert score > 0.0
    assert compute_answer_relevance("", answer) == 0.0


def test_evaluate_triad_supported():
    source_text = "Pydantic validates input schemas and serializes models."
    ev = [_make_evidence("S1", source_text, 0.9)]
    claim = Claim(
        source_id="S1",
        text="Pydantic validates input schemas.",
        quote="Pydantic validates input schemas and serializes models.",
    )
    metrics = evaluate_triad(
        query="What does Pydantic validate?",
        evidence=ev,
        answer_text="Pydantic validates input schemas. [S1]",
        claims=[claim],
        status="supported",
    )
    assert metrics.verdict in {"EXCELLENT", "GOOD"}
    assert metrics.groundedness > 0.8
    assert metrics.composite_score > 0.6


def test_evaluate_triad_abstained():
    metrics = evaluate_triad(
        query="Unknown concept?",
        evidence=[],
        answer_text="I could not find enough verified evidence.",
        claims=[],
        status="abstained",
    )
    assert metrics.verdict == "abstained"
    assert metrics.groundedness == 1.0
    assert metrics.composite_score == 0.0


def test_service_ask_includes_evaluation(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", provider="offline")
    service = KnowledgeService(settings)
    service.ingest("guide.txt", b"Atlas caches queries in memory for 60 seconds.")

    answer = service.ask(Question(text="How long does Atlas cache queries?"))
    assert answer.evaluation is not None
    assert answer.evaluation.groundedness >= 0.0
    assert answer.evaluation.composite_score >= 0.0
    assert answer.evaluation.verdict in {"EXCELLENT", "GOOD", "NEEDS_REVIEW", "abstained"}
