from __future__ import annotations

import math
from unittest.mock import Mock

from rag_assistant.config import Settings
from rag_assistant.hyde import (
    blend_vectors,
    extract_pseudo_relevance_terms,
    synthesize_hypothetical_document,
)
from rag_assistant.models import Question
from rag_assistant.providers import ModelClient
from rag_assistant.service import KnowledgeService


def test_synthesize_hypothetical_document_patterns():
    # How pattern
    doc_how = synthesize_hypothetical_document("How does Atlas handle Redis?")
    assert "Atlas" in doc_how
    assert "handle Redis" in doc_how
    assert "distributed" in doc_how

    # What is pattern
    doc_what = synthesize_hypothetical_document("What is Hermes?")
    assert "Hermes" in doc_what
    assert "architectural component" in doc_what

    # Why pattern
    doc_why = synthesize_hypothetical_document("Why does Atlas isolate tenants?")
    assert "Atlas" in doc_why
    assert "isolate tenants" in doc_why

    # Generic query
    doc_gen = synthesize_hypothetical_document("database failover strategies")
    assert "failover" in doc_gen or "database" in doc_gen


def test_blend_vectors():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    blended = blend_vectors(v1, v2, hyde_weight=0.5)

    # Norm should be 1.0
    norm = math.sqrt(sum(x * x for x in blended))
    assert abs(norm - 1.0) < 1e-4
    # Symmetrical weight
    assert abs(blended[0] - blended[1]) < 1e-4

    # Boundary weights
    assert blend_vectors(v1, v2, hyde_weight=0.0) == [1.0, 0.0]
    assert blend_vectors(v1, v2, hyde_weight=1.0) == [0.0, 1.0]

    # Edge cases
    assert blend_vectors([], [1.0]) == [1.0]
    assert blend_vectors([1.0], [1.0, 2.0]) == [1.0]


def test_extract_pseudo_relevance_terms():
    text = "Atlas orchestrates distributed microservices with automated failover and telemetry monitoring."
    query = "How does Atlas work?"
    terms = extract_pseudo_relevance_terms(text, query, max_terms=3)

    assert len(terms) <= 3
    # Terms must not include query words or stop words
    assert "atlas" not in terms
    assert "does" not in terms
    assert any(t in terms for t in ["orchestrates", "distributed", "microservices", "telemetry"])


def test_model_client_generate_hypothetical_document(tmp_path):
    settings = Settings(data_dir=tmp_path, provider="offline")
    client = ModelClient(settings)
    doc = client.generate_hypothetical_document("How does Atlas replicate data?")
    assert "Atlas" in doc
    assert len(doc) > 30

    # Mock remote client
    mock_settings = Settings(data_dir=tmp_path, provider="openai", model="mock-model")
    mock_client = ModelClient(mock_settings)
    mock_client._post = Mock(
        return_value={
            "choices": [{"message": {"content": "Synthetic passage about data replication."}}]
        }
    )
    doc_remote = mock_client.generate_hypothetical_document("How does Atlas replicate data?")
    assert doc_remote == "Synthetic passage about data replication."


def test_service_with_hyde_integration(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        provider="offline",
        hyde_enabled=True,
        hyde_weight=0.4,
    )
    service = KnowledgeService(settings)
    service.ingest(
        "system.md",
        (
            b"Atlas integrates with Redis for ultra-low latency distributed session persistence.\n"
            b"Redis memory mapping ensures high throughput under intense concurrent load.\n"
        ),
    )

    # Conceptual user question with vocabulary gap
    q = Question(text="What mechanisms provide fast session storage in Atlas?")
    ans = service.ask(q)

    assert ans.status == "supported"
    assert len(ans.evidence) >= 1
    assert any("Redis" in e.text for e in ans.evidence)
    assert len(ans.claims) >= 1
