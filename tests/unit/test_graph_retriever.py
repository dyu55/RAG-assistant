"""
Unit tests for the graph retriever (local + global search) and the
builder, using a fake Neo4jClient so no live database is needed.

The fake records every Cypher call so we can assert the right statements
are issued and returns pre-canned rows for each query.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from graph.builder import GraphBuilder
from graph.extractor import ExtractionResult, Entity, Relation
from graph.neo4j_client import Neo4jClient
from graph.retriever import GraphRetriever


class FakeNeo4j(Neo4jClient):
    """In-memory Neo4j double. Records reads/writes, replies with canned rows."""

    def __init__(self):
        # Skip parent __init__ — we don't want to actually open a driver.
        self.uri = "bolt://fake"
        self._driver = object()  # pretend we have one
        self._connected = True
        self.reads: list[tuple[str, dict | None]] = []
        self.writes: list[tuple[str, dict | None]] = []
        self._read_responses: list[list[dict]] = []
        self._write_responses: list[list[dict]] = []

    def ping(self) -> bool:
        return True

    def execute_read(self, cypher, parameters=None):
        self.reads.append((cypher, parameters))
        if self._read_responses:
            return self._read_responses.pop(0)
        return []

    def execute_write(self, cypher, parameters=None):
        self.writes.append((cypher, parameters))
        if self._write_responses:
            return self._write_responses.pop(0)
        return []

    def bootstrap_schema(self) -> None:
        pass


@pytest.fixture
def fake_neo4j():
    return FakeNeo4j()


@pytest.fixture
def mock_provider():
    p = Mock()
    p.generate_json.return_value = {
        "entities": [{"name": "GraphRAG", "type": "TECHNOLOGY"}],
        "relations": [],
    }
    return p


@pytest.fixture
def mock_extractor(mock_provider):
    """EntityRelationExtractor wrapping a mock provider."""
    from graph.extractor import EntityRelationExtractor
    return EntityRelationExtractor(mock_provider)


# ── GraphRetriever.local_search ───────────────────────────────────────────────

class TestGraphRetrieverLocalSearch:
    def test_local_search_extracts_entities_and_runs_cypher(self, fake_neo4j, mock_provider):
        # First execute_read is the entity-extraction sub-call inside
        # _subgraph_search. Return a couple of subgraph rows.
        fake_neo4j._read_responses = [
            [
                {
                    "focal_entity": "graphrag",
                    "path_weight": 3.0,
                    "path_entities": ["graphrag", "microsoft"],
                    "chunk_ids": ["c1", "c2"],
                    "filenames": ["paper.pdf"],
                },
                {
                    "focal_entity": "neo4j",
                    "path_weight": 2.0,
                    "path_entities": ["neo4j", "graphrag"],
                    "chunk_ids": ["c3"],
                    "filenames": ["paper.pdf"],
                },
            ]
        ]

        gr = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        chunks = gr.local_search("what does graphrag do?")

        assert len(chunks) == 2
        for c in chunks:
            assert c.retrieval_source == "graph"
            assert c.metadata["source"] == "graph"
        # Higher-weight entity should come first
        assert chunks[0].metadata["focal_entity"] == "graphrag"
        assert "graphrag" in chunks[0].text.lower()
        assert any("microsoft" in c.text.lower() for c in chunks)

        # The retriever should have issued exactly one cypher read.
        assert len(fake_neo4j.reads) == 1

    def test_local_search_falls_back_to_entity_overlap_when_no_entities(
        self, fake_neo4j, mock_provider
    ):
        # Provider returns NO entities → retriever falls back to text overlap.
        mock_provider.generate_json.return_value = {"entities": [], "relations": []}
        fake_neo4j._read_responses = [
            [
                {
                    "focal_entity": "neo4j",
                    "path_weight": 1.0,
                    "related_entities": ["graph"],
                    "predicates": ["USES"],
                    "chunk_ids": [],
                    "filenames": [],
                }
            ]
        ]

        gr = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        chunks = gr.local_search("tell me about neo4j")
        assert len(chunks) == 1
        assert chunks[0].retrieval_source == "graph"

    def test_local_search_empty_query_returns_empty(self, fake_neo4j, mock_provider):
        gr = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        assert gr.local_search("") == []
        assert gr.local_search("   ") == []


# ── GraphRetriever.global_search (map-reduce) ─────────────────────────────────

class TestGraphRetrieverGlobalSearch:
    def _communities(self):
        return [
            {
                "id": "L0-1",
                "level": 0,
                "title": "Microsoft GraphRAG",
                "summary": "Microsoft introduced GraphRAG for global summarization.",
                "findings": ["Improves global QA"],
                "key_entities": ["GraphRAG", "Microsoft"],
            },
            {
                "id": "L0-2",
                "level": 0,
                "title": "Neo4j for graphs",
                "summary": "Neo4j is a popular graph database.",
                "findings": ["Used by GraphRAG"],
                "key_entities": ["Neo4j"],
            },
        ]

    def test_global_search_runs_map_and_reduce(self, fake_neo4j, mock_provider):
        # First read: list communities.
        # Then for map: each community produces a partial answer.
        # Finally reduce: produces a single answer.
        fake_neo4j._read_responses = [self._communities()]
        mock_provider.generate_json.side_effect = [
            {"answer": "Microsoft GraphRAG helps global QA.", "relevance": 0.9,
             "reasoning": "directly relevant"},
            {"answer": "Neo4j stores the graph.", "relevance": 0.4,
             "reasoning": "tangential"},
            {"answer": "Combined: Microsoft GraphRAG improves global QA [Community 1] "
                       "and uses Neo4j [Community 2].",
             "reasoning": "merged", "cited": [1, 2]},
        ]

        gr = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        chunks = gr.global_search("summarize GraphRAG", one_shot=False)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.retrieval_source == "community"
        assert "GraphRAG" in chunk.text
        # Provider called 2 map + 1 reduce = 3 times
        assert mock_provider.generate_json.call_count == 3

    def test_global_search_returns_empty_when_no_communities(self, fake_neo4j, mock_provider):
        fake_neo4j._read_responses = [[]]
        gr = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        assert gr.global_search("anything") == []

    def test_global_search_partials_all_empty_returns_empty(self, fake_neo4j, mock_provider):
        fake_neo4j._read_responses = [self._communities()]
        mock_provider.generate_json.side_effect = [
            {"answer": "", "relevance": 0.0, "reasoning": "irrelevant"},
            {"answer": "", "relevance": 0.0, "reasoning": "irrelevant"},
        ]
        gr = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        assert gr.global_search("unrelated") == []


# ── GraphBuilder ──────────────────────────────────────────────────────────────

class TestGraphBuilder:
    def test_build_chunks_calls_upsert_per_chunk(self, fake_neo4j, mock_extractor):
        # First call: indexed_chunk_ids lookup → returns [] so all are processed.
        # Then one write per chunk.
        fake_neo4j._read_responses = [[]]  # nothing indexed yet

        builder = GraphBuilder(neo4j=fake_neo4j, extractor=mock_extractor)
        # Mock provider returns same entities for each chunk
        mock_extractor.provider.generate_json.return_value = {
            "entities": [
                {"name": "Alice", "type": "PERSON"},
                {"name": "Bob", "type": "PERSON"},
            ],
            "relations": [
                {"source": "Alice", "target": "Bob", "predicate": "KNOWS"}
            ],
        }

        chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "filename": "f.txt", "text": "Alice knows Bob."},
            {"chunk_id": "c2", "doc_id": "d1", "filename": "f.txt", "text": "More about Alice."},
        ]
        stats = builder.build_chunks(chunks)

        assert stats.chunks_processed == 2
        assert stats.chunks_skipped == 0
        assert stats.errors == 0
        # One write per chunk
        assert len(fake_neo4j.writes) == 2

    def test_build_chunks_skips_already_indexed(self, fake_neo4j, mock_extractor):
        fake_neo4j._read_responses = [[{"id": "c1"}]]  # c1 is already indexed

        builder = GraphBuilder(neo4j=fake_neo4j, extractor=mock_extractor)
        chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "filename": "f.txt", "text": "x"},
            {"chunk_id": "c2", "doc_id": "d1", "filename": "f.txt", "text": "y"},
        ]
        stats = builder.build_chunks(chunks)

        assert stats.chunks_skipped == 1
        assert stats.chunks_processed == 1
        assert len(fake_neo4j.writes) == 1

    def test_build_chunks_continues_on_failure(self, fake_neo4j, mock_extractor):
        fake_neo4j._read_responses = [[]]
        mock_extractor.extract_chunk = Mock(side_effect=Exception("always fails"))
        builder = GraphBuilder(neo4j=fake_neo4j, extractor=mock_extractor)
        chunks = [
            {"chunk_id": "c1", "doc_id": "d1", "filename": "f.txt", "text": "x"},
            {"chunk_id": "c2", "doc_id": "d1", "filename": "f.txt", "text": "y"},
        ]
        stats = builder.build_chunks(chunks)

        assert stats.chunks_processed == 0
        assert stats.errors == 2

    def test_build_chunks_empty_input(self, fake_neo4j, mock_extractor):
        builder = GraphBuilder(neo4j=fake_neo4j, extractor=mock_extractor)
        stats = builder.build_chunks([])
        assert stats.chunks_processed == 0
        # No Cypher calls
        assert fake_neo4j.reads == []
        assert fake_neo4j.writes == []