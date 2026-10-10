from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rag_assistant.config import Settings
from rag_assistant.late_chunking import (
    apply_late_chunking,
    blend_contextual_vector,
    compute_document_centroid,
    evaluate_late_chunking_retention,
)
from rag_assistant.models import Chunk, Question
from rag_assistant.service import KnowledgeService


def make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        document_id="doc1",
        filename="doc.txt",
        text=text,
        page=1,
        start=0,
        end=len(text),
    )


def test_compute_document_centroid():
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    v3 = [0.0, 0.0, 1.0]

    centroid = compute_document_centroid([v1, v2, v3])
    assert len(centroid) == 3
    # Check unit norm
    norm = math.sqrt(sum(v * v for v in centroid))
    assert pytest.approx(norm, 1e-6) == 1.0

    # Equal weight in each dimension
    assert centroid[0] == pytest.approx(centroid[1], 1e-6)
    assert centroid[1] == pytest.approx(centroid[2], 1e-6)

    # Empty list
    assert compute_document_centroid([]) == []

    # All zeros
    assert compute_document_centroid([[0.0, 0.0]]) == [0.0, 0.0]


def test_blend_contextual_vector():
    v_local = [1.0, 0.0]
    v_doc = [0.0, 1.0]

    # Blend with 20% weight
    v_late = blend_contextual_vector(v_local, v_doc, context_weight=0.20)
    assert len(v_late) == 2
    norm = math.sqrt(sum(v * v for v in v_late))
    assert pytest.approx(norm, 1e-6) == 1.0

    # Local component dominant, doc component present
    assert v_late[0] > v_late[1]
    assert v_late[1] > 0.0

    # Zero weight retains local
    v_zero = blend_contextual_vector(v_local, v_doc, context_weight=0.0)
    assert v_zero == v_local

    # Dimension mismatch returns local
    assert blend_contextual_vector(v_local, [1.0, 0.0, 0.0]) == v_local


def test_apply_late_chunking():
    chunks = [
        make_chunk("c1", "First chunk."),
        make_chunk("c2", "Second chunk."),
        make_chunk("c3", "Third chunk."),
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    late_vectors = apply_late_chunking(chunks, vectors, context_weight=0.25)
    assert len(late_vectors) == 3
    for lv in late_vectors:
        norm = math.sqrt(sum(v * v for v in lv))
        assert pytest.approx(norm, 1e-6) == 1.0

    # Single chunk edge case
    single_res = apply_late_chunking([chunks[0]], [vectors[0]], context_weight=0.25)
    assert single_res == [vectors[0]]

    # Zero weight edge case
    zero_weight_res = apply_late_chunking(chunks, vectors, context_weight=0.0)
    assert zero_weight_res == vectors


def test_evaluate_late_chunking_retention():
    v_chunk = [1.0, 0.0]
    v_doc = [0.0, 1.0]
    v_late = blend_contextual_vector(v_chunk, v_doc, context_weight=0.20)

    ret = evaluate_late_chunking_retention(v_chunk, v_doc, v_late)
    assert ret["local_fidelity"] > 0.95
    assert ret["context_gain"] > 0.15

    # Empty inputs
    assert evaluate_late_chunking_retention([], [], []) == {
        "local_fidelity": 1.0,
        "context_gain": 0.0,
    }


def test_service_late_chunking_end_to_end():
    with TemporaryDirectory() as tmp_dir:
        settings = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            late_chunking_enabled=True,
            late_chunking_weight=0.25,
            chunk_size=150,
            chunk_overlap=20,
        )
        service = KnowledgeService(settings)

        doc = (
            "Hypertable Database Specification.\n\n"
            "Hypertable is a high-performance distributed column database designed for massive write throughput.\n\n"
            "It manages data partitions using RangeServer processes distributed across cluster nodes.\n\n"
            "RangeServers commit mutation logs sequentially to durable append-only files."
        )
        service.ingest("hypertable.txt", doc.encode("utf-8"))

        # Query vector search specifically on the anaphoric second chunk
        answer = service.ask(
            Question(
                text="What is Hypertable and how do RangeServers commit mutations?",
                mode="vector",
                top_k=3,
            )
        )

        assert answer.status == "supported"
        assert len(answer.evidence) > 0
        assert any("RangeServer" in e.text for e in answer.evidence)

        # Test disabled late chunking setting
        settings_disabled = Settings(
            data_dir=Path(tmp_dir),
            provider="offline",
            embedding_provider="local",
            late_chunking_enabled=False,
            chunk_size=150,
            chunk_overlap=20,
        )
        service_disabled = KnowledgeService(settings_disabled)
        answer_disabled = service_disabled.ask(
            Question(
                text="What is Hypertable?",
                mode="vector",
                top_k=2,
            )
        )
        assert answer_disabled.status == "supported"
