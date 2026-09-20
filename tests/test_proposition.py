from __future__ import annotations

from rag_assistant.proposition import (
    Proposition,
    evaluate_proposition_overlap,
    extract_propositions,
)


def test_extract_propositions_empty():
    assert extract_propositions("") == []
    assert extract_propositions("   ") == []
    assert extract_propositions("Too short.") == []


def test_extract_propositions_compound_sentence():
    text = (
        "Atlas provides automatic failover mechanisms, "
        "and coordinates with Redis for distributed session caching."
    )
    props = extract_propositions(text)
    assert len(props) >= 2
    assert props[0].id == "P1"
    assert "Atlas provides automatic failover mechanisms" in props[0].text
    # Subject propagation: "Atlas" propagated into second clause
    assert "Atlas coordinates with Redis" in props[1].text
    assert props[0].entities or props[1].entities


def test_extract_propositions_semicolon_and_conjunctions():
    text = (
        "Vector indexing provides semantic search; "
        "BM25 provides exact keyword matching, as well as dense reranking improves precision."
    )
    props = extract_propositions(text)
    assert len(props) >= 2
    texts = [p.text for p in props]
    assert any("Vector indexing provides" in t for t in texts)
    assert any("BM25 provides" in t for t in texts)


def test_evaluate_proposition_overlap_empty():
    assert evaluate_proposition_overlap("", []) == 0.0
    assert evaluate_proposition_overlap("some claim", []) == 0.0
    p = Proposition(id="P1", text="Atlas uses Redis for caching.", entities=["Atlas", "Redis"])
    assert evaluate_proposition_overlap("", [p]) == 0.0
    assert evaluate_proposition_overlap("   ", [p]) == 0.0


def test_evaluate_proposition_overlap_matching():
    p1 = Proposition(
        id="P1", text="Atlas integrates with Redis for session cache.", entities=["Atlas", "Redis"]
    )
    p2 = Proposition(id="P2", text="Postgres manages transactional records.", entities=["Postgres"])
    props = [p1, p2]

    # Matching claim
    claim = "Atlas integrates with Redis cache"
    score = evaluate_proposition_overlap(claim, props)
    assert score > 0.7

    # Disjoint claim
    disjoint = "The weather today is sunny in Seattle"
    disjoint_score = evaluate_proposition_overlap(disjoint, props)
    assert disjoint_score == 0.0
