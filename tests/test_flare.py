"""
Unit and integration tests for FLARE (Forward-Looking Active REtrieval).
"""

from __future__ import annotations

from unittest.mock import Mock

from rag_assistant.config import Settings
from rag_assistant.flare import (
    detect_lookahead_uncertainty,
    execute_flare_active_retrieval,
    reground_claim_with_evidence,
    synthesize_active_query,
)
from rag_assistant.models import Claim, Draft, Evidence, Question
from rag_assistant.service import KnowledgeService


def _make_evidence(
    source_id: str,
    text: str,
    chunk_id: str = "chk_1",
    relevance: float = 0.85,
) -> Evidence:
    return Evidence(
        source_id=source_id,
        chunk_id=chunk_id,
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


def test_detect_lookahead_uncertainty():
    ev1 = _make_evidence("S1", "Atlas uses Redis for distributed caching and session persistence.")
    ev_map = {"S1": ev1}

    # Case 1: Missing source
    missing_source_claim = Claim(
        source_id="S99",
        text="Atlas integrates with Kafka message queues.",
        quote="Kafka message queues",
    )
    assert detect_lookahead_uncertainty(missing_source_claim, ev_map) is True

    # Case 2: Quote not in source
    bad_quote_claim = Claim(
        source_id="S1",
        text="Atlas uses Redis for caching.",
        quote="MongoDB is the primary database.",
    )
    assert detect_lookahead_uncertainty(bad_quote_claim, ev_map) is True

    # Case 3: Low overlap
    low_overlap_claim = Claim(
        source_id="S1",
        text="Kubernetes orchestrates containerized services in the cloud.",
        quote="Atlas uses Redis for distributed caching and session persistence.",
    )
    assert detect_lookahead_uncertainty(low_overlap_claim, ev_map, threshold=0.60) is True

    # Case 4: High overlap & valid quote
    valid_claim = Claim(
        source_id="S1",
        text="Atlas uses Redis for distributed caching.",
        quote="Atlas uses Redis for distributed caching and session persistence.",
    )
    assert detect_lookahead_uncertainty(valid_claim, ev_map, threshold=0.50) is False


def test_synthesize_active_query():
    # Test transition stripping and citation stripping
    query1 = synthesize_active_query(
        "Furthermore, Redis is deployed as a distributed cache cluster. [S1]"
    )
    assert "furthermore" not in query1.lower()
    assert "[S1]" not in query1
    assert "redis" in query1.lower()
    assert "cluster" in query1.lower()

    # Test stopwords removal
    query2 = synthesize_active_query("Additionally, the system can handle high throughput.")
    assert "additionally" not in query2.lower()
    assert "system" in query2.lower()
    assert "throughput" in query2.lower()

    # Fallback when minimal words
    query3 = synthesize_active_query("FastAPI")
    assert query3 == "FastAPI"


def test_reground_claim_with_evidence():
    claim = Claim(
        source_id="S_unknown",
        text="Atlas utilizes Raft consensus for leader election.",
        quote="initial placeholder quote",
    )

    ev_irrelevant = _make_evidence("S1", "Python 3.12 introduces performance improvements.")
    ev_relevant = _make_evidence(
        "S2",
        "Atlas utilizes Raft consensus for leader election and state synchronization across nodes.",
    )

    # Irrelevant candidate fails
    regrounded, matched_ev = reground_claim_with_evidence(claim, [ev_irrelevant])
    assert regrounded is None
    assert matched_ev is None

    # Relevant candidate succeeds
    regrounded, matched_ev = reground_claim_with_evidence(claim, [ev_irrelevant, ev_relevant])
    assert regrounded is not None
    assert matched_ev is not None
    assert regrounded.source_id == "S2"
    assert "Raft consensus" in regrounded.quote
    assert regrounded.text == claim.text


def test_execute_flare_active_retrieval_empty_or_grounded():
    ev1 = _make_evidence("S1", "Atlas provides sub-millisecond retrieval.")
    mock_retrieve = Mock(return_value=[])

    # Empty draft
    draft_empty = Draft(claims=[])
    d_out, ev_out, trace = execute_flare_active_retrieval(
        question="How fast is Atlas?",
        draft=draft_empty,
        evidence=[ev1],
        retrieve_fn=mock_retrieve,
    )
    assert len(d_out.claims) == 0
    assert trace.iterations == 0
    mock_retrieve.assert_not_called()

    # Grounded draft
    claim_grounded = Claim(
        source_id="S1",
        text="Atlas provides sub-millisecond retrieval.",
        quote="Atlas provides sub-millisecond retrieval.",
    )
    draft_grounded = Draft(claims=[claim_grounded])
    d_out2, ev_out2, trace2 = execute_flare_active_retrieval(
        question="How fast is Atlas?",
        draft=draft_grounded,
        evidence=[ev1],
        retrieve_fn=mock_retrieve,
        confidence_threshold=0.50,
    )
    assert len(d_out2.claims) == 1
    assert trace2.iterations == 0
    assert trace2.regrounded_claims_count == 0
    mock_retrieve.assert_not_called()


def test_execute_flare_active_retrieval_regrounds_uncertain_statement():
    ev_initial = _make_evidence("S1", "Atlas is a retrieval-augmented generation engine.")

    # Claim missing evidence
    uncertain_claim = Claim(
        source_id="S_fake",
        text="Atlas utilizes Raft consensus for leader election.",
        quote="unsupported placeholder quote",
    )
    draft = Draft(claims=[uncertain_claim])

    ev_supplementary = _make_evidence(
        source_id="S_new",
        chunk_id="chk_raft",
        text="Atlas utilizes Raft consensus for leader election.",
        relevance=0.92,
    )

    def mock_retrieve_fn(q: Question) -> list[Evidence]:
        assert "raft" in q.text.lower() or "consensus" in q.text.lower()
        return [ev_supplementary]

    d_out, ev_out, trace = execute_flare_active_retrieval(
        question="What consensus protocol does Atlas use?",
        draft=draft,
        evidence=[ev_initial],
        retrieve_fn=mock_retrieve_fn,
        max_lookaheads=2,
    )

    assert trace.iterations == 1
    assert trace.regrounded_claims_count == 1
    assert len(trace.active_retrievals) == 1
    assert trace.active_retrievals[0].grounded is True
    assert len(ev_out) == 2  # initial + supplementary
    assert d_out.claims[0].source_id == "S2"
    assert "Raft consensus" in d_out.claims[0].quote


def test_execute_flare_active_retrieval_unresolved_lookahead():
    ev_initial = _make_evidence("S1", "Atlas is a retrieval-augmented generation engine.")

    uncertain_claim = Claim(
        source_id="S_missing",
        text="Atlas has quantum computing acceleration modules.",
        quote="quantum placeholder quote",
    )
    draft = Draft(claims=[uncertain_claim])

    # Retriever returns nothing or irrelevant evidence
    def mock_retrieve_fn(q: Question) -> list[Evidence]:
        return []

    d_out, ev_out, trace = execute_flare_active_retrieval(
        question="Does Atlas support quantum computing?",
        draft=draft,
        evidence=[ev_initial],
        retrieve_fn=mock_retrieve_fn,
        max_lookaheads=1,
    )

    assert trace.iterations == 1
    assert trace.regrounded_claims_count == 0
    assert len(trace.active_retrievals) == 1
    assert trace.active_retrievals[0].grounded is False
    assert len(ev_out) == 1


def test_flare_service_end_to_end_integration():
    settings = Settings(
        provider="offline",
        flare_enabled=True,
        flare_max_lookaheads=2,
        self_rag_enabled=True,
    )
    service = KnowledgeService(settings)

    # Ingest two documents
    service.ingest(
        filename="service_a.txt",
        content=b"Atlas Core handles inverted indexing and BM25 ranking across shards.",
    )
    service.ingest(
        filename="service_b.txt",
        content=b"Atlas Cluster implements Raft consensus for leader election and state synchronization.",
    )

    # Ask question
    answer = service.ask(
        Question(
            text="How does Atlas Core handle indexing and what consensus protocol does the cluster use?",
            mode="keyword",
        )
    )

    assert answer.status == "supported"
    assert answer.flare is not None
    assert answer.flare.enabled is True
