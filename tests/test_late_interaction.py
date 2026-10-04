from __future__ import annotations

from rag_assistant.late_interaction import (
    compute_maxsim,
    late_interaction_score,
    token_similarity,
)
from rag_assistant.models import Chunk
from rag_assistant.rerank import compute_rerank_score


def test_token_similarity():
    assert token_similarity("redis", "redis") == 1.0
    # Common prefix root match
    assert token_similarity("replicate", "replication") >= 0.85
    assert token_similarity("cluster", "clustering") >= 0.85
    # Completely disjoint
    assert token_similarity("apple", "banana") == 0.0


def test_compute_maxsim_exact_and_partial():
    # Exact match across all query terms
    query = "Atlas Redis cache"
    doc_exact = "Atlas coordinates with Redis cache cluster for fast lookups."
    score_exact, alignments = compute_maxsim(query, doc_exact)
    assert score_exact == 1.0
    assert alignments["atlas"] == ("atlas", 1.0)
    assert alignments["redis"] == ("redis", 1.0)
    assert alignments["cache"] == ("cache", 1.0)

    # Partial match (2 out of 3 tokens matched)
    doc_partial = "Atlas uses Redis for session storage."
    score_part, part_alignments = compute_maxsim(query, doc_partial)
    assert 0.60 <= score_part <= 0.75
    assert part_alignments["cache"][1] == 0.0  # 'cache' missing


def test_compute_maxsim_empty_and_disjoint():
    assert compute_maxsim("", "Some text")[0] == 0.0
    assert compute_maxsim("Query", "")[0] == 0.0

    score_disjoint, _ = compute_maxsim(
        "Atlas Redis", "Astronomers observe distant galaxy clusters."
    )
    assert score_disjoint == 0.0


def test_late_interaction_chunk_and_rerank_integration():
    c_relevant = Chunk(
        id="c1",
        document_id="d1",
        filename="system.md",
        text="Atlas cluster architecture coordinates automatic failover across nodes.",
        page=1,
        start=0,
        end=75,
        entities=["Atlas"],
    )
    c_irrelevant = Chunk(
        id="c2",
        document_id="d1",
        filename="system.md",
        text="The quick brown fox jumps over the lazy sleeping dog.",
        page=1,
        start=76,
        end=130,
        entities=[],
    )

    score_rel = late_interaction_score("Atlas automatic failover", c_relevant)
    score_irrel = late_interaction_score("Atlas automatic failover", c_irrelevant)

    assert score_rel >= 0.90
    assert score_irrel == 0.0

    # Rerank score with Late Interaction MaxSim enabled
    rerank_val = compute_rerank_score(
        "Atlas automatic failover",
        c_relevant,
        vector_score=0.8,
        use_late_interaction=True,
    )
    assert rerank_val >= 0.75
