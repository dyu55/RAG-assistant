"""Tests for fast-path routing, one-shot global search, and graph-augmented chunks."""

from __future__ import annotations

import time
from unittest.mock import Mock

from graph.retriever import GraphRetriever
from graph.router import QueryRouter, RouteMode
from ingestion.chunker import Chunk


class TestFastPathRouter:
    """Verify that QueryRouter's fast-path bypasses LLM latency on obvious queries."""

    def test_global_keywords_route_without_llm(self):
        mock_provider = Mock()
        router = QueryRouter(provider=mock_provider, mode_setting="auto")

        t0 = time.perf_counter()
        decision = router.route("Please summarize the main themes across all documents")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Must execute in <10ms and make 0 LLM calls
        assert elapsed_ms < 50
        assert mock_provider.generate_json.call_count == 0
        assert decision.mode == RouteMode.GLOBAL
        assert decision.confidence >= 0.8
        assert "Fast-path" in decision.reason

    def test_both_comparison_keywords_route_without_llm(self):
        mock_provider = Mock()
        router = QueryRouter(provider=mock_provider, mode_setting="auto")

        decision = router.route("Compare and contrast the architectures of System A and System B")
        assert mock_provider.generate_json.call_count == 0
        assert decision.mode == RouteMode.BOTH
        assert "Fast-path" in decision.reason

    def test_chinese_global_keywords_route_without_llm(self):
        mock_provider = Mock()
        router = QueryRouter(provider=mock_provider, mode_setting="auto")

        decision = router.route("总结所有文档的核心主题与架构大纲")
        assert mock_provider.generate_json.call_count == 0
        assert decision.mode == RouteMode.GLOBAL

    def test_ambiguous_query_falls_back_to_llm(self):
        mock_provider = Mock()
        mock_provider.generate_json.return_value = {
            "mode": "LOCAL",
            "confidence": 0.95,
            "reason": "specific entity",
        }
        router = QueryRouter(provider=mock_provider, mode_setting="auto")

        decision = router.route("Tell me about the performance benchmark numbers")
        assert mock_provider.generate_json.call_count == 1
        assert decision.mode == RouteMode.LOCAL


class TestOneShotGlobalSearch:
    """Verify that One-Shot Global Search synthesizes answers in a single LLM call."""

    def _mock_communities(self):
        return [
            {
                "id": "c1",
                "level": 0,
                "title": "Auth Architecture",
                "summary": "Handles JWT authentication and token refresh.",
                "findings": ["Uses RS256 signing"],
                "key_entities": ["AuthService", "JWT"],
            },
            {
                "id": "c2",
                "level": 0,
                "title": "Database Architecture",
                "summary": "PostgreSQL database with read replicas.",
                "findings": ["Sub-10ms read latency"],
                "key_entities": ["PostgreSQL", "ReplicaPool"],
            },
        ]

    def test_one_shot_global_search_makes_single_llm_call(self):
        fake_neo4j = Mock()
        fake_neo4j.execute_read.return_value = self._mock_communities()

        mock_provider = Mock()
        mock_provider.generate_json.return_value = {
            "answer": "The system uses JWT for auth [Community 1] and PostgreSQL for storage [Community 2].",
            "reasoning": "Synthesized from communities 1 and 2",
            "cited": [1, 2],
        }

        retriever = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        chunks = retriever.global_search("Summarize system architecture", one_shot=True)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.retrieval_source == "community"
        assert "JWT" in chunk.text
        # Only 1 LLM call in one-shot mode (instead of 2 map + 1 reduce = 3 calls)
        assert mock_provider.generate_json.call_count == 1
        assert chunk.metadata.get("one_shot") is True

    def test_fallback_to_map_reduce_when_one_shot_fails(self):
        fake_neo4j = Mock()
        fake_neo4j.execute_read.return_value = self._mock_communities()

        mock_provider = Mock()
        # 1st call (one-shot) fails with empty answer; next calls run map & reduce
        mock_provider.generate_json.side_effect = [
            {"answer": "", "reasoning": "failed"},
            {"answer": "Partial 1", "relevance": 0.8, "reasoning": "ok"},
            {"answer": "Partial 2", "relevance": 0.8, "reasoning": "ok"},
            {"answer": "Final reduced answer.", "reasoning": "merged", "cited": [1, 2]},
        ]

        retriever = GraphRetriever(neo4j=fake_neo4j, provider=mock_provider)
        chunks = retriever.global_search("Summarize system architecture", one_shot=True)

        assert len(chunks) == 1
        assert "Final reduced answer" in chunks[0].text


class TestGraphAugmentedChunks:
    """Verify Chunk.with_graph_context adds structured entity/relation tags."""

    def test_with_graph_context_formatting(self):
        chunk = Chunk(
            text="Tesla delivered the first Cybertruck in late 2023.",
            chunk_id="c123",
            doc_id="d1",
            index=0,
        )
        augmented = chunk.with_graph_context(
            entities=["Tesla", "Cybertruck", "Elon Musk"],
            relations=["Tesla -(PRODUCES)-> Cybertruck"],
        )

        assert "[Entities: Tesla, Cybertruck, Elon Musk]" in augmented
        assert "[Relations: Tesla -(PRODUCES)-> Cybertruck]" in augmented
        assert "Tesla delivered the first Cybertruck" in augmented

    def test_with_graph_context_empty_returns_plain_text(self):
        chunk = Chunk(
            text="Plain text without entities.",
            chunk_id="c124",
            doc_id="d1",
            index=1,
        )
        assert chunk.with_graph_context([], []) == chunk.text
