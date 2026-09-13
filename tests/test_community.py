from __future__ import annotations

from fastapi.testclient import TestClient

from rag_assistant.api import create_app
from rag_assistant.community import (
    build_entity_graph,
    detect_communities,
    global_community_retrieve,
    is_global_query,
    retrieve_communities,
)
from rag_assistant.config import Settings
from rag_assistant.models import Chunk, Question
from rag_assistant.service import KnowledgeService


def test_build_entity_graph():
    c1 = Chunk(
        id="c1",
        document_id="d1",
        filename="doc1.txt",
        text="Atlas integrates with Redis for fast caching.",
        page=1,
        start=0,
        end=45,
        entities=["Atlas", "Redis"],
    )
    c2 = Chunk(
        id="c2",
        document_id="d1",
        filename="doc1.txt",
        text="Atlas also connects to Postgres database.",
        page=1,
        start=46,
        end=88,
        entities=["Atlas", "Postgres"],
    )
    entity_to_chunks, neighbors, lookup = build_entity_graph([c1, c2])

    assert "atlas" in entity_to_chunks
    assert "c1" in entity_to_chunks["atlas"] and "c2" in entity_to_chunks["atlas"]
    assert neighbors["atlas"]["redis"] == 1
    assert neighbors["atlas"]["postgres"] == 1
    assert "c1" in lookup and "c2" in lookup


def test_detect_communities_empty():
    assert detect_communities([]) == []

    chunk_without_entities = Chunk(
        id="c0",
        document_id="d0",
        filename="empty.txt",
        text="Plain text without any capitalized entities.",
        page=1,
        start=0,
        end=45,
        entities=[],
    )
    assert detect_communities([chunk_without_entities]) == []


def test_detect_communities_two_distinct_clusters():
    # Cluster A: In-memory store
    cA1 = Chunk(
        id="ca1",
        document_id="da",
        filename="cache.md",
        text="Atlas coordinates with Redis cluster. Redis stores distributed sessions securely.",
        page=1,
        start=0,
        end=82,
        entities=["Atlas", "Redis"],
    )
    cA2 = Chunk(
        id="ca2",
        document_id="da",
        filename="cache.md",
        text="Redis provides memory-mapped persistence for Atlas workflows.",
        page=1,
        start=83,
        end=145,
        entities=["Atlas", "Redis"],
    )

    # Cluster B: Streaming pipeline
    cB1 = Chunk(
        id="cb1",
        document_id="db",
        filename="stream.md",
        text="Kafka brokers ingest event streams. Flink operators process Kafka topic records.",
        page=1,
        start=0,
        end=80,
        entities=["Kafka", "Flink"],
    )
    cB2 = Chunk(
        id="cb2",
        document_id="db",
        filename="stream.md",
        text="Flink pipelines output enriched telemetry from Kafka topics.",
        page=1,
        start=81,
        end=141,
        entities=["Kafka", "Flink"],
    )

    communities = detect_communities([cA1, cA2, cB1, cB2])
    assert len(communities) == 2

    titles = [c.title for c in communities]
    assert any("Atlas" in t or "Redis" in t for t in titles)
    assert any("Kafka" in t or "Flink" in t for t in titles)

    for comm in communities:
        assert len(comm.hub_entities) >= 1
        assert len(comm.chunk_ids) == 2
        assert len(comm.summary) > 10
        assert comm.weight >= 1


def test_is_global_query():
    assert is_global_query("Summarize all the components in the architecture") is True
    assert is_global_query("Give me a high-level overview of all systems") is True
    assert is_global_query("Architecture summary across all documents") is True
    assert is_global_query("What are the main themes?") is True
    assert is_global_query("What port does Redis listen on?") is False
    assert is_global_query("How does authentication validate tokens?") is False


def test_retrieve_communities():
    cA = Chunk(
        id="ca",
        document_id="da",
        filename="c.md",
        text="Atlas engine utilizes Redis caching.",
        page=1,
        start=0,
        end=36,
        entities=["Atlas", "Redis"],
    )
    cB = Chunk(
        id="cb",
        document_id="db",
        filename="s.md",
        text="Kafka brokers publish telemetry.",
        page=1,
        start=0,
        end=32,
        entities=["Kafka", "Broker"],
    )
    communities = detect_communities([cA, cB])

    matched = retrieve_communities("Redis cache details", communities, top_k=1)
    assert len(matched) == 1
    assert "Redis" in matched[0].hub_entities or "Atlas" in matched[0].hub_entities

    fallback = retrieve_communities("completely unknown query", communities, top_k=1)
    assert len(fallback) == 1


def test_global_community_retrieve():
    c1 = Chunk(
        id="c1",
        document_id="d1",
        filename="sys.md",
        text="Atlas gateway orchestrates backend microservices. High availability is guaranteed.",
        page=1,
        start=0,
        end=82,
        entities=["Atlas", "Gateway"],
    )
    c2 = Chunk(
        id="c2",
        document_id="d2",
        filename="db.md",
        text="Postgres stores customer records. Read replicas handle reporting queries.",
        page=1,
        start=0,
        end=73,
        entities=["Postgres", "Replicas"],
    )
    q = Question(text="Summarize all key systems and architecture", mode="graph")
    evidence = global_community_retrieve(q, [c1, c2], top_k=2)

    assert len(evidence) >= 1
    ev = evidence[0]
    assert "community" in ev.channels
    assert "[Community:" in ev.context_window
    assert ev.relevance >= 0.8


def test_api_graph_communities_endpoint(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", provider="offline")
    service = KnowledgeService(settings)
    service.ingest("arch.txt", b"Atlas interacts with Redis. Redis provides caching for Atlas.")

    app = create_app(settings, service)
    with TestClient(app) as client:
        res = client.get("/api/graph/communities")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "Atlas" in data[0]["hub_entities"] or "Redis" in data[0]["hub_entities"]
