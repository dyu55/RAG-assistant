from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag_assistant.config import Settings
from rag_assistant.models import Chunk, Question
from rag_assistant.raptor import (
    build_raptor_layers,
    cluster_chunks,
    compute_token_jaccard,
    compute_vector_similarity,
    expand_with_raptor_tree,
    synthesize_cluster_summary,
)
from rag_assistant.service import KnowledgeService


def make_test_chunk(
    cid: str,
    text: str,
    doc_id: str = "doc1",
    vector: list[float] | None = None,
    entities: list[str] | None = None,
) -> Chunk:
    return Chunk(
        id=cid,
        document_id=doc_id,
        filename="test_doc.txt",
        text=text,
        page=1,
        start=0,
        end=len(text),
        vector=vector or [],
        entities=entities or [],
    )


def test_compute_vector_and_token_similarity():
    # Identical unit vectors
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    assert compute_vector_similarity(v1, v2) == 1.0

    # Orthogonal vectors
    v3 = [0.0, 1.0]
    assert compute_vector_similarity(v1, v3) == 0.0

    # Dimension mismatch or empty
    assert compute_vector_similarity(v1, [1.0, 0.0, 0.0]) == 0.0
    assert compute_vector_similarity([], []) == 0.0

    # Token Jaccard
    assert (
        compute_token_jaccard("distributed system architecture", "distributed system design") > 0.4
    )
    assert compute_token_jaccard("apple banana", "car truck") == 0.0
    assert compute_token_jaccard("", "something") == 0.0


def test_cluster_chunks():
    assert cluster_chunks([]) == []

    c1 = make_test_chunk("c1", "Microservices communicate via REST or gRPC APIs.")
    c2 = make_test_chunk("c2", "Services expose RESTful endpoints and gRPC contracts.")
    c3 = make_test_chunk("c3", "Kubernetes pods manage container lifecycles in nodes.")
    c4 = make_test_chunk("c4", "Container runtimes execute pods within Kubernetes clusters.")

    # With <= cluster_size
    assert len(cluster_chunks([c1, c2], cluster_size=4)) == 1

    # Clustering based on token similarity
    clusters = cluster_chunks([c1, c2, c3, c4], cluster_size=2)
    assert len(clusters) >= 2


def test_synthesize_cluster_summary():
    v1 = [0.6, 0.8]
    v2 = [0.8, 0.6]
    c1 = make_test_chunk(
        "c1",
        "Kafka partition replicas provide fault tolerance against broker outages.",
        vector=v1,
        entities=["Kafka", "Broker"],
    )
    c2 = make_test_chunk(
        "c2",
        "Consumer groups read partition offsets committed by coordinator brokers.",
        vector=v2,
        entities=["Kafka", "Consumer"],
    )

    summary_chunk = synthesize_cluster_summary([c1, c2], layer=1)
    assert summary_chunk.id.startswith("raptor:L1:")
    assert "[RAPTOR L1 Summary]" in summary_chunk.text
    assert "Kafka" in summary_chunk.entities
    assert "Consumer" in summary_chunk.entities
    assert len(summary_chunk.vector) == 2

    # Check L2 norm of centroid vector is approximately 1.0
    norm = math.sqrt(sum(v * v for v in summary_chunk.vector))
    assert pytest.approx(norm, 1e-6) == 1.0


def test_build_raptor_layers_and_expand():
    # Create 6 chunks
    chunks = [
        make_test_chunk(
            f"c{i}",
            f"Passage {i}: Distributed databases replicate state across consensus clusters.",
            vector=[1.0 if i % 2 == 0 else 0.0, 1.0 if i % 2 == 1 else 0.0],
            entities=[f"Entity_{i}"],
        )
        for i in range(6)
    ]

    tree = build_raptor_layers(chunks, max_layers=2, cluster_size=3)
    assert 0 in tree
    assert len(tree[0]) == 6
    assert 1 in tree
    assert len(tree[1]) >= 1

    expanded, meta = expand_with_raptor_tree(chunks, max_layers=2, cluster_size=3)
    assert len(expanded) > len(chunks)
    assert all(cid in meta for cid in [n.id for n in tree[1]])
    assert all(meta[n.id] == 1 for n in tree[1])

    # Edge cases
    assert len(build_raptor_layers([chunks[0]], max_layers=2)[0]) == 1
    assert len(build_raptor_layers([], max_layers=2)[0]) == 0


def test_service_raptor_end_to_end():
    with TemporaryDirectory() as tmp_dir:
        settings = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            raptor_enabled=True,
            raptor_max_layers=2,
            raptor_cluster_size=2,
            chunk_size=150,
            chunk_overlap=20,
        )
        service = KnowledgeService(settings)

        doc_content = (
            "Apache Cassandra is a highly scalable distributed NoSQL database designed for high availability.\n\n"
            "Cassandra uses peer-to-peer gossip protocols across masterless cluster nodes.\n\n"
            "Data is partitioned using consistent hashing tokens across a ring topology.\n\n"
            "Tunable consistency levels allow developers to balance latency versus linearizable safety."
        )
        service.ingest("cassandra_architecture.txt", doc_content.encode("utf-8"))

        # Query asking for high-level architectural overview
        answer = service.ask(
            Question(
                text="What is the architecture and consistency model of Cassandra?",
                mode="keyword",
                top_k=4,
            )
        )
        assert answer.status == "supported"
        assert answer.raptor_nodes_count >= 1
        assert len(answer.evidence) > 0

        # Query with raptor_enabled=False
        settings_disabled = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            raptor_enabled=False,
            chunk_size=150,
            chunk_overlap=20,
        )
        service_disabled = KnowledgeService(settings_disabled)
        answer_disabled = service_disabled.ask(
            Question(
                text="What is Cassandra?",
                mode="keyword",
                top_k=2,
            )
        )
        assert answer_disabled.status == "supported"
        assert answer_disabled.raptor_nodes_count == 0
