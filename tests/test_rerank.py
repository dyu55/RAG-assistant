from __future__ import annotations

from rag_assistant.models import Chunk, Question
from rag_assistant.rerank import (
    analyze_query_weights,
    compute_rerank_score,
    score_entity_alignment,
    score_phrase_match,
    score_term_proximity,
)
from rag_assistant.retrieval import retrieve


def test_analyze_query_weights():
    # Quoted identifier boosts keyword
    w_quote = analyze_query_weights('"exact phrase" query')
    assert w_quote["keyword"] > w_quote["vector"]
    assert round(sum(w_quote.values()), 1) == 3.0

    # Relational entity intent boosts graph
    w_graph = analyze_query_weights("What is the relationship between Atlas and Hermes?")
    assert w_graph["graph"] >= w_graph["keyword"]
    assert round(sum(w_graph.values()), 1) == 3.0

    # Long conceptual question boosts vector
    w_vector = analyze_query_weights(
        "Why does distributed consensus require an odd number of quorum voters in Raft?"
    )
    assert w_vector["vector"] > 1.0
    assert round(sum(w_vector.values()), 1) == 3.0


def test_score_phrase_match():
    query = "distributed cache"
    exact_text = "We deploy a distributed cache cluster across three regions."
    assert score_phrase_match(query, exact_text) == 1.0

    three_word_query = "distributed cache cluster"
    partial_text = "The distributed cache is deployed across regions."
    assert 0.0 < score_phrase_match(three_word_query, partial_text) < 1.0

    disjoint_text = "Relational databases manage transactions with strict ACID guarantees."
    assert score_phrase_match(query, disjoint_text) == 0.0


def test_score_term_proximity():
    query_terms = {"distributed", "cache"}
    tight_text = "High-availability distributed cache infrastructure."
    loose_text = (
        "Distributed systems are complex. Many components exist. Networks fail. "
        "Disks wear out. Finally, we discuss how to manage cache."
    )
    score_tight = score_term_proximity(query_terms, tight_text)
    score_loose = score_term_proximity(query_terms, loose_text)
    assert score_tight > score_loose
    assert score_term_proximity(set(), "Some text") == 0.0


def test_score_entity_alignment():
    query = "How does Atlas interact with Kubernetes?"
    assert score_entity_alignment(query, ["Atlas", "Redis"], []) >= 0.7
    assert score_entity_alignment(query, ["Storage"], ["Atlas", "Kubernetes"]) == 0.75
    assert score_entity_alignment("generic question without entities", ["Atlas"], []) == 0.5
    assert score_entity_alignment(query, ["Unrelated"], []) == 0.2


def test_compute_rerank_score():
    chunk = Chunk(
        id="c1",
        document_id="d1",
        filename="arch.md",
        text="Atlas utilizes distributed cache for ultra low latency lookups.",
        page=1,
        start=0,
        end=64,
        entities=["Atlas"],
    )
    score_high = compute_rerank_score(
        "Atlas distributed cache",
        chunk,
        vector_score=0.9,
        graph_score=1.0,
        path=["Atlas"],
    )
    assert score_high >= 0.7

    disjoint_chunk = Chunk(
        id="c2",
        document_id="d1",
        filename="arch.md",
        text="Astronomers discovered distant exoplanets orbiting binary stars.",
        page=1,
        start=65,
        end=128,
        entities=[],
    )
    score_zero = compute_rerank_score(
        "Atlas distributed cache",
        disjoint_chunk,
        vector_score=0.0,
        graph_score=0.0,
    )
    assert score_zero == 0.0


def test_retrieve_includes_rerank_and_context_window():
    c1 = Chunk(
        id="c1",
        document_id="d1",
        filename="guide.md",
        text="Overview of Atlas microservices.",
        page=1,
        start=0,
        end=32,
        entities=["Atlas"],
    )
    c2 = Chunk(
        id="c2",
        document_id="d1",
        filename="guide.md",
        text="Atlas integrates with Redis for session persistence.",
        page=1,
        start=33,
        end=85,
        entities=["Atlas", "Redis"],
    )
    q = Question(text="How does Atlas integrate with Redis?", mode="hybrid", top_k=2)
    evidence = retrieve(q, [c1, c2], vector=None)

    assert len(evidence) >= 1
    top = evidence[0]
    assert top.chunk_id == "c2"
    assert top.rerank_score > 0.4
    assert top.context_window != ""
    assert "Atlas integrates with Redis" in top.context_window
