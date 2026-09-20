from __future__ import annotations

from rag_assistant.config import Settings
from rag_assistant.hierarchical import (
    ParentChunk,
    _create_parent,
    build_parent_chunks,
    rollup_to_parent_context,
)
from rag_assistant.models import Chunk, Evidence, Question
from rag_assistant.service import KnowledgeService


def test_build_parent_chunks_empty():
    parents, mapping = build_parent_chunks([])
    assert parents == {}
    assert mapping == {}


def test_build_parent_chunks_single_doc():
    c1 = Chunk(
        id="c1",
        document_id="doc1",
        filename="guide.md",
        page=1,
        start=0,
        end=50,
        text="Atlas architecture employs distributed indexing.",
    )
    c2 = Chunk(
        id="c2",
        document_id="doc1",
        filename="guide.md",
        page=1,
        start=40,
        end=95,
        text="distributed indexing. Nodes query local shards efficiently.",
    )
    chunks = [c1, c2]

    parents, child_to_parent = build_parent_chunks(chunks, max_parent_size=500)
    assert len(parents) == 1
    p_id = next(iter(parents))
    parent = parents[p_id]
    assert parent.document_id == "doc1"
    assert parent.filename == "guide.md"
    assert parent.page == 1
    assert parent.start == 0
    assert parent.end == 95
    assert "Atlas architecture employs distributed indexing." in parent.text
    assert "Nodes query local shards efficiently." in parent.text
    assert child_to_parent["c1"] == p_id
    assert child_to_parent["c2"] == p_id


def test_build_parent_chunks_max_size_split():
    c1 = Chunk(
        id="c1",
        document_id="doc1",
        filename="guide.md",
        page=1,
        start=0,
        end=100,
        text="A" * 80,
    )
    c2 = Chunk(
        id="c2",
        document_id="doc1",
        filename="guide.md",
        page=1,
        start=100,
        end=200,
        text="B" * 80,
    )
    chunks = [c1, c2]

    parents, child_to_parent = build_parent_chunks(chunks, max_parent_size=100)
    assert len(parents) == 2
    assert child_to_parent["c1"] != child_to_parent["c2"]


def test_create_parent_stitching_and_overlap():
    c1 = Chunk(
        id="c1",
        document_id="doc1",
        filename="paper.pdf",
        page=2,
        start=0,
        end=29,
        text="Hierarchical chunking enables",
    )
    c2 = Chunk(
        id="c2",
        document_id="doc1",
        filename="paper.pdf",
        page=2,
        start=22,
        end=53,
        text="enables small-to-big retrieval.",
    )
    parent = _create_parent([c1, c2], doc_id="doc1", page=2)
    assert parent.start == 0
    assert parent.end == 53
    assert "Hierarchical chunking enables" in parent.text
    assert "small-to-big retrieval." in parent.text
    assert parent.child_ids == ["c1", "c2"]


def test_rollup_to_parent_context():
    parent = ParentChunk(
        id="p1",
        document_id="doc1",
        filename="system.md",
        page=1,
        start=0,
        end=150,
        text="Full section text containing broad background and specific metrics.",
        child_ids=["c1"],
    )
    parents = {"p1": parent}
    child_to_parent = {"c1": "p1"}

    ev = Evidence(
        source_id="S1",
        chunk_id="c1",
        document_id="doc1",
        filename="system.md",
        page=1,
        start=30,
        end=80,
        text="specific metrics are tracked.",
        relevance=0.88,
        fusion_score=0.1,
        channels=["keyword"],
    )

    enriched = rollup_to_parent_context([ev], parents, child_to_parent)
    assert len(enriched) == 1
    assert enriched[0].text == ev.text
    assert "[Parent Section: system.md p.1]" in enriched[0].context_window
    assert parent.text in enriched[0].context_window


def test_rollup_to_parent_context_edge_cases():
    assert rollup_to_parent_context([], {}, {}) == []

    ev = Evidence(
        source_id="S1",
        chunk_id="unknown_child",
        document_id="doc1",
        filename="system.md",
        page=1,
        start=0,
        end=10,
        text="test",
        relevance=0.5,
        fusion_score=0.1,
        channels=["keyword"],
    )
    result = rollup_to_parent_context([ev], {}, {"c1": "p1"})
    assert result[0] == ev


def test_hierarchical_end_to_end_service(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        provider="offline",
        chunk_size=150,
        chunk_overlap=30,
    )
    service = KnowledgeService(settings)
    doc_content = (
        b"Section 1: Distributed Vector Indexing.\n"
        b"Atlas maintains localized vector shards across multiple storage nodes.\n"
        b"Each shard can execute approximate nearest neighbor search independently.\n"
        b"Section 2: Query Routing and Aggregation.\n"
        b"Central coordinator aggregates candidate vectors and reranks them.\n"
    )
    service.ingest("distributed.txt", doc_content)

    q = Question(text="How does Atlas maintain localized vector shards?")
    answer = service.ask(q)

    assert answer.status == "supported"
    assert len(answer.evidence) >= 1
    for ev in answer.evidence:
        assert ev.context_window != ""
        assert "[Parent Section: distributed.txt" in ev.context_window
        assert ev.text.strip() in ev.context_window
