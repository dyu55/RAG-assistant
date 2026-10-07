from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag_assistant.config import Settings
from rag_assistant.models import Chunk, Question
from rag_assistant.mrl import (
    determine_adaptive_coarse_dim,
    evaluate_mrl_retention,
    mrl_funnel_vector_scores,
    truncate_and_normalize,
)
from rag_assistant.retrieval import vector_scores
from rag_assistant.service import KnowledgeService


def make_unit_vector(dim: int, active_idx: int) -> list[float]:
    vec = [0.0] * dim
    vec[active_idx % dim] = 1.0
    return vec


def test_truncate_and_normalize():
    vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    sub = truncate_and_normalize(vec, 3)
    assert len(sub) == 3
    # Check L2 norm equals 1.0
    norm = math.sqrt(sum(v * v for v in sub))
    assert pytest.approx(norm, 1e-6) == 1.0

    # Zero vector handling
    zero_vec = [0.0] * 5
    assert truncate_and_normalize(zero_vec, 3) == [0.0, 0.0, 0.0]

    # Target dimension <= 0 raises ValueError
    with pytest.raises(ValueError, match="Target dimension must be positive"):
        truncate_and_normalize(vec, 0)


def test_determine_adaptive_coarse_dim():
    assert determine_adaptive_coarse_dim(1536) == 128
    assert determine_adaptive_coarse_dim(768) == 64
    assert determine_adaptive_coarse_dim(384) == 64
    assert determine_adaptive_coarse_dim(128) == 32
    assert determine_adaptive_coarse_dim(64) == 64


def test_mrl_funnel_vector_scores_empty_and_errors():
    assert mrl_funnel_vector_scores([1.0, 0.0], []) == {}

    chunk_bad = Chunk(
        id="c1",
        document_id="d1",
        filename="f.txt",
        text="text",
        page=1,
        start=0,
        end=4,
        vector=[1.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="Index embedding dimensions changed"):
        mrl_funnel_vector_scores([1.0, 0.0], [chunk_bad])


def test_mrl_funnel_vector_scores_filtering():
    dim = 128
    q_vec = make_unit_vector(dim, 2)

    chunks = []
    # Create 80 chunks; c0 is identical to query, c1-c9 are close, rest are orthogonal
    for i in range(80):
        if i == 0:
            c_vec = make_unit_vector(dim, 2)
        elif i < 10:
            # Shared energy in leading dimensions
            raw = [0.0] * dim
            raw[2] = 0.8
            raw[i % 5] = 0.6
            c_vec = truncate_and_normalize(raw, dim)
        else:
            c_vec = make_unit_vector(dim, 50 + (i % 50))
        chunks.append(
            Chunk(
                id=f"c{i}",
                document_id="d1",
                filename="f.txt",
                text=f"Chunk {i} content text.",
                page=1,
                start=0,
                end=20,
                vector=c_vec,
            )
        )

    # Coarse pool size of 15
    scores = mrl_funnel_vector_scores(
        q_vec, chunks, coarse_dim=32, candidate_pool_size=15, blend_alpha=0.85
    )

    # c0 should be the highest scored chunk
    assert "c0" in scores
    assert scores["c0"] > 0.8
    # Only top 15 candidates should be returned
    assert len(scores) <= 15


def test_evaluate_mrl_retention():
    dim = 64
    q_vec = make_unit_vector(dim, 0)
    chunks = [
        Chunk(
            id=f"c{i}",
            document_id="d1",
            filename="f.txt",
            text=f"Sample text {i}",
            page=1,
            start=0,
            end=10,
            vector=make_unit_vector(dim, i % 10),
        )
        for i in range(20)
    ]

    retention = evaluate_mrl_retention(q_vec, chunks, coarse_dim=16, top_k=5)
    assert 0.0 <= retention["recall_at_k"] <= 1.0
    assert retention["coarse_dim"] == 16.0
    assert retention["full_dim"] == 64.0
    assert retention["compression_ratio"] == 0.25

    # Empty chunks
    empty_ret = evaluate_mrl_retention(q_vec, [], coarse_dim=16)
    assert empty_ret["recall_at_k"] == 1.0


def test_vector_scores_with_mrl_integration():
    dim = 128
    q_vec = make_unit_vector(dim, 1)

    chunks = [
        Chunk(
            id=f"chk_{i}",
            document_id="doc1",
            filename="doc.txt",
            text=f"Documentation excerpt {i}",
            page=1,
            start=0,
            end=20,
            vector=make_unit_vector(dim, 1 if i == 0 else (10 + i % 100)),
        )
        for i in range(70)
    ]

    # When use_mrl is True and len(chunks) > candidate_pool (60)
    scores_mrl = vector_scores(
        q_vec,
        chunks,
        use_mrl=True,
        coarse_dim=32,
        candidate_pool=25,
        blend_alpha=0.9,
    )
    assert "chk_0" in scores_mrl
    assert len(scores_mrl) <= 25

    # When use_mrl is False
    scores_direct = vector_scores(q_vec, chunks, use_mrl=False)
    assert "chk_0" in scores_direct
    assert scores_direct["chk_0"] == 1.0


def test_service_mrl_end_to_end():
    with TemporaryDirectory() as tmp_dir:
        settings = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            mrl_enabled=True,
            mrl_coarse_dim=64,
            mrl_candidate_pool=30,
        )
        service = KnowledgeService(settings)

        doc_content = (
            "PostgreSQL is a powerful, open-source object-relational database system.\n\n"
            "PostgreSQL uses Multi-Version Concurrency Control (MVCC) to ensure ACID compliance.\n\n"
            "Write-Ahead Logging (WAL) guarantees data integrity and enables streaming replication."
        )
        service.ingest("postgres_architecture.txt", doc_content.encode("utf-8"))

        # Query using vector retrieval mode
        ans_vector = service.ask(
            Question(
                text="How does PostgreSQL handle concurrency and replication?",
                mode="vector",
                top_k=3,
            )
        )
        assert ans_vector.status == "supported"
        assert len(ans_vector.evidence) > 0

        # Query using hybrid retrieval mode
        ans_hybrid = service.ask(
            Question(
                text="What is PostgreSQL MVCC?",
                mode="hybrid",
                top_k=3,
            )
        )
        assert ans_hybrid.status == "supported"
        assert len(ans_hybrid.evidence) > 0
