"""
Unit tests for Corrective RAG (CRAG) and Self-RAG inference-time correction guardrail.
"""

from __future__ import annotations

from unittest.mock import Mock

from rag_assistant.config import Settings
from rag_assistant.correction import (
    correct_query_for_fallback,
    is_retrieval_insufficient,
    rescue_unsupported_draft,
)
from rag_assistant.models import Claim, Draft, Evidence, Question
from rag_assistant.service import KnowledgeService, verify


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


def test_is_retrieval_insufficient():
    assert is_retrieval_insufficient([]) is True
    assert is_retrieval_insufficient([_make_evidence("S1", "low relevance", 0.05)]) is True
    assert is_retrieval_insufficient([_make_evidence("S1", "high relevance", 0.65)]) is False


def test_correct_query_for_fallback():
    assert "atlas cache" in correct_query_for_fallback(
        "Could you please explain how Atlas cache operates?"
    )
    assert "indexing strategy" in correct_query_for_fallback("Tell me about indexing strategy?")
    assert correct_query_for_fallback("FastAPI routing") == "fastapi routing"


def test_rescue_unsupported_draft_already_valid():
    ev = [_make_evidence("S1", "FastAPI handles asynchronous web routing with high throughput.")]
    valid_draft = Draft(
        claims=[
            Claim(
                source_id="S1",
                text="FastAPI handles asynchronous web routing.",
                quote="FastAPI handles asynchronous web routing with high throughput.",
            )
        ]
    )
    draft, is_rescued, reason = rescue_unsupported_draft("FastAPI routing", valid_draft, ev, verify)
    assert not is_rescued
    assert draft == valid_draft
    assert "All quoted spans were located" in reason


def test_rescue_unsupported_draft_rescues_hallucination():
    source_text = "Atlas stores customer profiles with AES-256 encryption."
    ev = [_make_evidence("S1", source_text)]

    # Fabricated quote and unsupported claim
    hallucinated_draft = Draft(
        claims=[
            Claim(
                source_id="S1",
                text="Atlas uses plain text storage for passwords.",
                quote="Fabricated text that does not exist.",
            )
        ]
    )
    draft, is_rescued, reason = rescue_unsupported_draft(
        "How does Atlas store profiles?", hallucinated_draft, ev, verify
    )
    assert is_rescued is True
    assert "Rescued" in reason
    assert len(draft.claims) >= 1
    # Rescued claims are grounded in actual source text
    assert draft.claims[0].quote in source_text


def test_rescue_unsupported_draft_fails_when_no_evidence():
    hallucinated_draft = Draft(
        claims=[
            Claim(
                source_id="S1",
                text="Some claim.",
                quote="Missing quote.",
            )
        ]
    )
    draft, is_rescued, _ = rescue_unsupported_draft("query", hallucinated_draft, [], verify)
    assert not is_rescued


def test_service_ask_self_correction_workflow(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", provider="openai", model="gpt-4o-mini")
    mock_client = Mock()
    mock_client.embed.return_value = [[0.1] * 384]

    # Mock model generating an ungrounded hallucination
    mock_client.generate.return_value = Draft(
        claims=[
            Claim(
                source_id="S1",
                text="Hallucinated text without basis.",
                quote="Fake quote that does not appear anywhere.",
            )
        ]
    )

    service = KnowledgeService(settings, client=mock_client)
    service.ingest(
        "doc.txt", b"Atlas supports multi-tenant workload isolation and high availability."
    )

    answer = service.ask(Question(text="How does Atlas support multi-tenancy?"))

    # Verify self-correction successfully rescued the ungrounded draft
    assert answer.status == "supported"
    assert "rescued_unsupported_draft_via_extractive_fallback" in answer.corrections
    assert any("Initial draft failed strict citation verification" in w for w in answer.warnings)
