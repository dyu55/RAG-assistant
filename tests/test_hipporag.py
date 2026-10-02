from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.hipporag import (
    compute_personalized_pagerank,
    find_shortest_associative_path,
    hipporag_graph_scores,
)
from rag_assistant.models import Chunk, Question
from rag_assistant.service import KnowledgeService


def test_compute_personalized_pagerank_triangle():
    # Triangle graph: A - B, B - C, C - A
    adjacency = {
        "A": {"B": 1.0, "C": 1.0},
        "B": {"A": 1.0, "C": 1.0},
        "C": {"A": 1.0, "B": 1.0},
    }
    # Personalization vector concentrated on A
    p = {"A": 1.0}
    ppr = compute_personalized_pagerank(adjacency, p, damping=0.85)

    assert len(ppr) == 3
    # A must have the highest stationary probability
    assert ppr["A"] > ppr["B"]
    assert ppr["A"] > ppr["C"]
    # B and C are symmetric
    assert round(ppr["B"], 4) == round(ppr["C"], 4)
    # Sum of probabilities equals 1.0
    assert round(sum(ppr.values()), 3) == 1.0


def test_compute_personalized_pagerank_empty():
    assert compute_personalized_pagerank({}, {}) == {}
    assert compute_personalized_pagerank({"A": {}}, {"A": 0.0}) == {}


def test_find_shortest_associative_path():
    adj = {
        "atlas": {"beacon": 1.0},
        "beacon": {"atlas": 1.0, "cobalt": 1.0},
        "cobalt": {"beacon": 1.0, "delta": 1.0},
        "delta": {"cobalt": 1.0},
    }
    # Target is seed
    assert find_shortest_associative_path(["atlas"], "atlas", adj) == ["atlas"]
    # Multi-hop
    assert find_shortest_associative_path(["atlas"], "cobalt", adj) == ["atlas", "beacon", "cobalt"]
    assert find_shortest_associative_path(["atlas"], "delta", adj) == [
        "atlas",
        "beacon",
        "cobalt",
        "delta",
    ]


def test_hipporag_graph_scores_associative_decay():
    c1 = Chunk(
        id="c1",
        document_id="d1",
        filename="1.txt",
        text="Atlas coordinates directly with Beacon gateway.",
        page=1,
        start=0,
        end=45,
        entities=["Atlas", "Beacon"],
    )
    c2 = Chunk(
        id="c2",
        document_id="d2",
        filename="2.txt",
        text="Beacon routes requests to Cobalt storage engine.",
        page=1,
        start=0,
        end=48,
        entities=["Beacon", "Cobalt"],
    )
    c3 = Chunk(
        id="c3",
        document_id="d3",
        filename="3.txt",
        text="Cobalt archives historical snapshots to Delta lake.",
        page=1,
        start=0,
        end=52,
        entities=["Cobalt", "Delta"],
    )
    chunks = [c1, c2, c3]

    scores, paths = hipporag_graph_scores("Atlas", chunks, damping=0.85)

    assert "c1" in scores and "c2" in scores and "c3" in scores
    # Scores decay monotonically along associative hops
    assert scores["c1"] > scores["c2"]
    assert scores["c2"] > scores["c3"]
    # Multi-hop path traced for c3
    assert paths["c3"] == ["atlas", "beacon", "cobalt"]


def test_hipporag_graph_scores_empty_inputs():
    assert hipporag_graph_scores("", []) == ({}, {})
    c = Chunk(
        id="c1",
        document_id="d1",
        filename="1.txt",
        text="Plain text.",
        page=1,
        start=0,
        end=10,
        entities=[],
    )
    assert hipporag_graph_scores("query", [c]) == ({}, {})


def test_service_hipporag_end_to_end(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        provider="offline",
        graph_algorithm="ppr",
        ppr_damping=0.85,
    )
    service = KnowledgeService(settings)
    c1_txt = b"Atlas manages distributed coordination with Beacon service.\n"
    c2_txt = b"Beacon routes transactions safely to Cobalt database engine.\n"
    service.ingest("arch1.txt", c1_txt)
    service.ingest("arch2.txt", c2_txt)

    q = Question(text="How does Atlas interact with Beacon and Cobalt?", mode="graph")
    answer = service.ask(q)

    assert answer.status == "supported"
    assert len(answer.evidence) >= 1
    assert any("Beacon" in e.text for e in answer.evidence)
    # Valid verified claims
    assert len(answer.claims) >= 1
    for claim in answer.claims:
        assert any(claim.quote in e.text for e in answer.evidence)
