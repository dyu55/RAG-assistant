"""
Unit tests for adaptive retrieval router and dynamic mode selection.
"""

from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.models import Question
from rag_assistant.router import route_query
from rag_assistant.service import KnowledgeService


def test_route_empty_query():
    decision = route_query("")
    assert decision.mode == "hybrid"
    assert decision.confidence == 1.0


def test_route_relational_intent_to_graph():
    decision = route_query("What entities are connected to LangChain in the knowledge base?")
    assert decision.mode == "graph"
    assert decision.confidence > 0.8
    assert "entity relationships" in decision.reasoning.lower()


def test_route_exact_identifiers_to_keyword():
    # Short query with specific uppercase constant / code / number
    decision = route_query("Error HTTP_404 code 500")
    assert decision.mode == "keyword"
    assert "identifiers" in decision.reasoning.lower()


def test_route_quoted_phrases_to_keyword():
    decision = route_query('"exact phrase citation"')
    assert decision.mode == "keyword"


def test_route_conceptual_intent_to_vector():
    decision = route_query("Explain the high-level philosophy and conceptual idea behind attention")
    assert decision.mode == "vector"
    assert "semantic" in decision.reasoning.lower()


def test_route_general_query_to_hybrid():
    decision = route_query(
        "How does chunking improve document indexing performance in RAG systems?"
    )
    assert decision.mode == "hybrid"
    assert decision.confidence >= 0.9


def test_service_ask_with_mode_auto(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", provider="offline")
    service = KnowledgeService(settings)
    content = (
        b"FastAPI provides high-performance asynchronous web routing. "
        b"Pydantic validates input schemas and serializes models."
    )
    service.ingest("web.txt", content)

    # Ask with mode="auto"
    question = Question(text="How does FastAPI provide routing?", mode="auto")
    answer = service.ask(question)

    assert answer.status in {"supported", "abstained"}
    assert answer.retrieval_mode in {"hybrid", "keyword", "vector", "graph"}
    assert any("Adaptive routing selected" in w for w in answer.warnings)
