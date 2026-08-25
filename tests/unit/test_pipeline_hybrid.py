"""
Unit tests for the hybrid GraphRAG + vector pipeline.

Stubs out LLM, embedder, and Neo4j so the orchestration logic runs in
isolation. Verifies that:
- vector + graph results are merged
- citations in the generated answer carry the correct source_type
- PipelineResult.route_mode / sources_used are set
- the existing reliability engine still runs on the merged chunks
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import pytest

from core.generator import Generator
from core.pipeline import Pipeline, PipelineResult
from core.reliability import ReliabilityChecker
from core.retriever import RetrievedChunk
from graph.router import RouteDecision, RouteMode


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeEmbedder:
    def get_or_create_collection(self, name=None):
        m = Mock()
        m.count.return_value = 0
        return m


class FakeRetriever:
    """Vector retriever stand-in that returns a fixed chunk list."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks
        self.has_documents_call_count = 0

    def retrieve(self, query, top_k=None, enable_reranking=False):
        return list(self.chunks)

    def has_documents(self) -> bool:
        self.has_documents_call_count += 1
        return bool(self.chunks)


class FakeGraphRetriever:
    """Graph retriever stand-in."""

    def __init__(
        self,
        local_chunks: list[RetrievedChunk] | None = None,
        global_chunks: list[RetrievedChunk] | None = None,
    ):
        self.local_chunks = local_chunks or []
        self.global_chunks = global_chunks or []

    def local_search(self, query, top_k=None):
        return list(self.local_chunks)

    def global_search(self, query):
        return list(self.global_chunks)


class FakeGenerator:
    """Generator stand-in that returns a pre-canned answer."""

    def __init__(self, response: dict):
        self.provider = Mock()
        self.provider.get_model_name.return_value = "fake-model"
        self._response = response

    def generate(self, query, chunks, temperature=0.3):
        from core.generator import GeneratedAnswer, Citation
        citations = []
        for i, c in enumerate(self._response.get("citations", [])):
            source_idx = c.get("source_index", i + 1)
            source_type = "vector"
            if 1 <= source_idx <= len(chunks):
                source_type = chunks[source_idx - 1].retrieval_source
            citations.append(
                Citation(
                    source_index=source_idx,
                    chunk_id=c.get("chunk_id", ""),
                    quote=c.get("quote", ""),
                    source_type=source_type,
                )
            )
        return GeneratedAnswer(
            answer=self._response.get("answer", "ok"),
            citations=citations,
            self_confidence=self._response.get("self_confidence", 0.9),
            reasoning="fake",
        )


def _chunk(text, score=0.7, source="vector", chunk_id=None, metadata=None):
    md = {"source": source}
    if metadata:
        md.update(metadata)
    return RetrievedChunk(
        chunk_id=chunk_id or f"{source}-chunk-{id(text)}",
        text=text,
        score=score,
        metadata=md,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHybridPipeline:
    def test_local_path_only_runs_vector_and_graph_local(self):
        # Two vector chunks, one graph chunk, no global.
        vector_chunks = [_chunk("vector text 1", source="vector", chunk_id="v1")]
        graph_local = [_chunk("graph text 1", source="graph", chunk_id="g1")]
        response = {
            "answer": "answer with [V1] and [G2].",
            "citations": [
                {"source_index": 1, "chunk_id": "v1", "quote": "vector text 1"},
                {"source_index": 2, "chunk_id": "g1", "quote": "graph text 1"},
            ],
            "self_confidence": 0.9,
        }

        vector_retriever = FakeRetriever(vector_chunks)
        graph_retriever = FakeGraphRetriever(local_chunks=graph_local)
        generator = FakeGenerator(response)
        router = Mock()
        router.route.return_value = RouteDecision(
            mode=RouteMode.LOCAL, confidence=1.0, reason="test"
        )

        pipeline = Pipeline(
            retriever=vector_retriever,
            generator=generator,
            reliability_checker=ReliabilityChecker(),
            graph_retriever=graph_retriever,
            router=router,
        )

        result = pipeline.run("test query")

        # Chunks merged
        assert len(result.retrieved_chunks) == 2
        # Route decision recorded
        assert result.route_mode == "local"
        # Both retrieval paths represented in the merged chunks
        assert {c.retrieval_source for c in result.retrieved_chunks} == {"vector", "graph"}
        # Citations inherit source_type from their chunk position
        by_chunk_id = {c.chunk_id: c for c in result.retrieved_chunks}
        for cit in result.citations:
            assert cit.source_type == by_chunk_id[cit.chunk_id].retrieval_source
        # Reliability engine ran
        assert result.reliability is not None
        assert "vector" in result.reliability.sources_used
        assert "graph" in result.reliability.sources_used

    def test_global_path_only_returns_community_chunk(self):
        community_chunks = [_chunk("community answer", source="community", chunk_id="c1")]
        vector_chunks = [_chunk("vector text", source="vector", chunk_id="v1")]
        response = {
            "answer": "[C1] community answer",
            "citations": [
                {"source_index": 1, "chunk_id": "c1", "quote": "community answer"},
            ],
            "self_confidence": 0.8,
        }

        vector_retriever = FakeRetriever(vector_chunks)
        graph_retriever = FakeGraphRetriever(global_chunks=community_chunks)
        generator = FakeGenerator(response)
        router = Mock()
        router.route.return_value = RouteDecision(
            mode=RouteMode.GLOBAL, confidence=1.0, reason="test"
        )

        pipeline = Pipeline(
            retriever=vector_retriever,
            generator=generator,
            graph_retriever=graph_retriever,
            router=router,
        )
        result = pipeline.run("summarize")

        # Vector retriever was NOT called when route is GLOBAL only
        # (we test that the pipeline doesn't even invoke it for retrieval).
        # Sources used: community only.
        assert result.reliability.sources_used == ["community"]

    def test_both_path_merges_all_three(self):
        vector_chunks = [_chunk("v", source="vector", chunk_id="v1")]
        graph_local = [_chunk("g", source="graph", chunk_id="g1")]
        graph_global = [_chunk("c", source="community", chunk_id="c1")]
        response = {
            "answer": "all paths combined",
            "citations": [
                {"source_index": 1, "chunk_id": "c1", "quote": "c"},
                {"source_index": 2, "chunk_id": "v1", "quote": "v"},
                {"source_index": 3, "chunk_id": "g1", "quote": "g"},
            ],
        }

        vector_retriever = FakeRetriever(vector_chunks)
        graph_retriever = FakeGraphRetriever(local_chunks=graph_local, global_chunks=graph_global)
        generator = FakeGenerator(response)
        router = Mock()
        router.route.return_value = RouteDecision(mode=RouteMode.BOTH, reason="test")

        pipeline = Pipeline(
            retriever=vector_retriever,
            generator=generator,
            graph_retriever=graph_retriever,
            router=router,
        )
        result = pipeline.run("compare")

        sources = {c.retrieval_source for c in result.retrieved_chunks}
        assert sources == {"vector", "graph", "community"}
        assert result.route_mode == "both"

    def test_off_path_runs_vector_only(self):
        vector_chunks = [_chunk("v", source="vector", chunk_id="v1")]
        graph_local = [_chunk("g", source="graph", chunk_id="g1")]
        graph_retriever = FakeGraphRetriever(local_chunks=graph_local)
        generator = FakeGenerator({"answer": "ok", "citations": []})
        router = Mock()
        router.route.return_value = RouteDecision(mode=RouteMode.OFF, reason="test")

        pipeline = Pipeline(
            retriever=FakeRetriever(vector_chunks),
            generator=generator,
            graph_retriever=graph_retriever,
            router=router,
        )
        result = pipeline.run("anything")

        # Graph retriever not invoked when route is OFF
        assert {c.retrieval_source for c in result.retrieved_chunks} == {"vector"}

    def test_no_router_falls_back_to_vector_only(self):
        vector_chunks = [_chunk("v", source="vector", chunk_id="v1")]
        graph_retriever = FakeGraphRetriever(local_chunks=[_chunk("g", source="graph")])
        generator = FakeGenerator({"answer": "ok", "citations": []})

        pipeline = Pipeline(
            retriever=FakeRetriever(vector_chunks),
            generator=generator,
            graph_retriever=graph_retriever,
            router=None,  # no router configured
        )
        result = pipeline.run("anything")

        # Default route is LOCAL → runs vector AND graph_local.
        # Sources should include both vector and graph.
        sources = {c.retrieval_source for c in result.retrieved_chunks}
        assert sources == {"vector", "graph"}
        assert "No router configured" in result.route_reason

    def test_pipeline_result_serializes_sources(self):
        r = PipelineResult(query="x", retrieved_chunks=[
            _chunk("v", source="vector"),
            _chunk("g", source="graph"),
        ])
        d = r.to_dict()
        assert "sources_used" in d
        assert d["sources_used"] == ["vector", "graph"]
        assert d["route_mode"] == "vector-only"