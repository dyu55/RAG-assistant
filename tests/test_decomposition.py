"""
Unit tests for query decomposition and faceted sub-query retrieval.
"""

from __future__ import annotations

from rag_assistant.decomposition import (
    decompose_query,
    faceted_retrieve,
    is_compound_query,
)
from rag_assistant.models import Chunk, Question


def test_is_compound_query():
    assert is_compound_query("What is A? How does B work?")
    assert is_compound_query("Compare vector search with keyword search")
    assert is_compound_query("Explain difference between dense and sparse embeddings")
    assert is_compound_query("What is chunking and also how does caching operate")
    assert not is_compound_query("What is chunking?")
    assert not is_compound_query("Explain the RAG pipeline.")


def test_decompose_query_atomic():
    assert decompose_query("What is ChromaDB?") == ["What is ChromaDB?"]
    assert decompose_query("") == []
    assert decompose_query("   ") == []


def test_decompose_query_multiple_sentences():
    res = decompose_query("What is chunking? How does deduplication work?")
    assert len(res) == 2
    assert "What is chunking?" in res[0]
    assert "How does deduplication work?" in res[1]


def test_decompose_query_comparative():
    res = decompose_query("Compare Vector Retrieval and Graph Retrieval")
    assert len(res) == 2
    assert "Vector Retrieval" in res[0]
    assert "Graph Retrieval" in res[1]

    res_vs = decompose_query("BM25 vs Dense Embeddings")
    assert len(res_vs) == 2
    assert "BM25" in res_vs[0]
    assert "Dense Embeddings" in res_vs[1]


def test_decompose_query_conjunction_clauses():
    res = decompose_query("How does indexing work and what is the cache size?")
    assert len(res) == 2
    assert "indexing work" in res[0]
    assert "cache size" in res[1]


def test_faceted_retrieve_single_query():
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            filename="doc.txt",
            text="Vector indexing stores high-dimensional embeddings.",
            page=1,
            start=0,
            end=50,
            vector=[0.5, 0.5],
            entities=["Vector"],
        )
    ]
    q = Question(text="Vector indexing", mode="keyword")
    evidence = faceted_retrieve(q, chunks)
    assert len(evidence) == 1
    assert evidence[0].chunk_id == "c1"


def test_faceted_retrieve_compound_query():
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            filename="doc1.txt",
            text="Vector indexing stores high-dimensional embeddings.",
            page=1,
            start=0,
            end=50,
            vector=[0.9, 0.1],
            entities=["Vector"],
        ),
        Chunk(
            id="c2",
            document_id="d2",
            filename="doc2.txt",
            text="Graph retrieval traverses entity relations in knowledge graphs.",
            page=1,
            start=0,
            end=60,
            vector=[0.1, 0.9],
            entities=["Graph"],
        ),
    ]
    q = Question(
        text="What is vector indexing? How does graph retrieval operate?", top_k=2, mode="keyword"
    )
    evidence = faceted_retrieve(q, chunks)
    # Both facets should be represented in the merged evidence
    assert len(evidence) == 2
    chunk_ids = {e.chunk_id for e in evidence}
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
