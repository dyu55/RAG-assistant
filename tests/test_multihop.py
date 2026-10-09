from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from rag_assistant.config import Settings
from rag_assistant.models import Chunk, Evidence, Question
from rag_assistant.multihop import (
    detect_multihop_need,
    execute_multihop_search,
    extract_bridge_entities,
    synthesize_followup_query,
)
from rag_assistant.service import KnowledgeService


def make_evidence_item(
    sid: str,
    cid: str,
    text: str,
    relevance: float = 0.8,
) -> Evidence:
    return Evidence(
        source_id=sid,
        chunk_id=cid,
        document_id="doc1",
        filename="test.txt",
        text=text,
        page=1,
        start=0,
        end=len(text),
        relevance=relevance,
        fusion_score=0.5,
        channels=["hybrid"],
    )


def test_detect_multihop_need():
    # Multi-hop queries with relational dependencies
    assert detect_multihop_need(
        "What caching store does Atlas use and how does it handle failover?"
    )
    assert detect_multihop_need(
        "Which database powers the catalog and what is its replication model?"
    )
    assert detect_multihop_need("What framework is behind the API gateway and why does it restart?")

    # Simple / single-hop queries
    assert not detect_multihop_need("What is Redis?")
    assert not detect_multihop_need("Explain how replication works.")
    assert not detect_multihop_need("Short text")


def test_extract_bridge_entities():
    query = "What caching store does Atlas use and how does it handle failover?"
    ev1 = make_evidence_item(
        "S1",
        "c1",
        "Atlas implements Redis Cluster for high-throughput session state caching.",
    )
    chunk1 = Chunk(
        id="c1",
        document_id="doc1",
        filename="doc.txt",
        text=ev1.text,
        page=1,
        start=0,
        end=len(ev1.text),
        entities=["Redis Cluster", "Atlas"],
    )

    bridges = extract_bridge_entities(query, [ev1], [chunk1])
    assert "Redis Cluster" in bridges
    # "Atlas" was already in the query, so it shouldn't be the bridge
    assert "Atlas" not in bridges

    # Empty evidence
    assert extract_bridge_entities(query, []) == []


def test_synthesize_followup_query():
    query = "What caching store does Atlas use and how does it handle failover?"
    bridges = ["Redis Cluster"]
    followup = synthesize_followup_query(query, bridges)
    assert "Redis Cluster" in followup
    assert "failover" in followup.lower()

    # Pronoun replacement
    query2 = "Atlas uses a store and its replication is distributed across zones."
    followup2 = synthesize_followup_query(query2, ["Cassandra"])
    assert "Cassandra" in followup2

    # Empty bridges
    assert synthesize_followup_query("query", []) == "query"


def test_execute_multihop_search_mocked():
    ev_hop1 = make_evidence_item(
        "S1",
        "c_atlas",
        "Atlas uses Redis for low-latency in-memory session caching.",
        relevance=0.88,
    )
    ev_hop2 = make_evidence_item(
        "S1",
        "c_redis",
        "Redis Sentinel provides automated failover and master election.",
        relevance=0.92,
    )

    q = Question(text="What caching store does Atlas use and how does it handle failover?")
    chunk1 = Chunk(
        id="c_atlas",
        document_id="d1",
        filename="f1.txt",
        text=ev_hop1.text,
        page=1,
        start=0,
        end=len(ev_hop1.text),
        entities=["Redis", "Atlas"],
    )

    def mock_retriever(sub_q: Question) -> list[Evidence]:
        if "redis" in sub_q.text.lower() and "failover" in sub_q.text.lower():
            return [ev_hop2]
        return [ev_hop1]

    final_evidence, trace = execute_multihop_search(
        question=q,
        chunks=[chunk1],
        retrieve_fn=mock_retriever,
        max_hops=2,
    )

    assert trace.hops_executed == 2
    assert "Redis" in trace.bridge_entities
    assert len(trace.sub_queries) == 2
    assert len(final_evidence) == 2
    # Verify provenance paths
    paths = [p for e in final_evidence for p in e.path]
    assert "multihop:hop_1" in paths
    assert "multihop:hop_2" in paths


def test_service_multihop_end_to_end():
    with TemporaryDirectory() as tmp_dir:
        settings = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            multihop_enabled=True,
            multihop_max_hops=2,
            chunk_size=150,
            chunk_overlap=20,
        )
        service = KnowledgeService(settings)

        doc1 = (
            "PaymentGateway Service Architecture.\n\n"
            "The PaymentGateway system uses MongoDB for customer transaction ledgers.\n\n"
            "Transactions are signed with asymmetric HMAC keys before persistence."
        )
        doc2 = (
            "MongoDB Reliability Overview.\n\n"
            "MongoDB replica sets achieve high availability using automated Raft election consensus.\n\n"
            "Secondary members replicate the primary oplog asynchronously."
        )
        service.ingest("payment.txt", doc1.encode("utf-8"))
        service.ingest("mongo.txt", doc2.encode("utf-8"))

        # Relational multi-hop query
        q = Question(
            text="What database does PaymentGateway use and how does it achieve high availability?",
            mode="keyword",
            top_k=4,
        )
        answer = service.ask(q)

        assert answer.status == "supported"
        assert answer.multihop is not None
        assert answer.multihop.hops_executed == 2
        assert any("mongodb" in b.lower() for b in answer.multihop.bridge_entities)
        assert any("multihop_iterative_retrieval_executed" in c for c in answer.corrections)

        # Test disabled setting
        settings_disabled = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            multihop_enabled=False,
            chunk_size=150,
            chunk_overlap=20,
        )
        service_disabled = KnowledgeService(settings_disabled)
        answer_disabled = service_disabled.ask(
            Question(
                text="What database does PaymentGateway use?",
                mode="keyword",
                top_k=2,
            )
        )
        assert answer_disabled.status == "supported"
        assert answer_disabled.multihop is None
