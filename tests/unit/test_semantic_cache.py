"""
Unit tests for core/cache.py (SemanticCache & CacheStats).
"""

from __future__ import annotations

import time

from core.cache import SemanticCache, _cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_similarity_is_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors_similarity_is_zero(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        assert _cosine_similarity(v1, v2) == 0.0

    def test_empty_or_mismatched_length(self):
        assert _cosine_similarity([], [1.0]) == 0.0
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0


class TestSemanticCache:
    def test_exact_match_tier1_hit(self):
        cache = SemanticCache()
        payload = {"answer": "GraphRAG merges vectors and knowledge graphs.", "citations": []}
        cache.put("What is GraphRAG?", [0.1, 0.2], payload)

        res, score, hit_type = cache.get("what is graphrag?")
        assert res == payload
        assert score == 1.0
        assert hit_type == "exact"
        assert cache.stats.exact_hits == 1

    def test_semantic_match_tier2_hit(self):
        cache = SemanticCache(similarity_threshold=0.90)
        payload = {"answer": "Transformer attention is parallelizable."}
        # Embedding for stored query
        cache.put("How does Transformer attention work?", [1.0, 0.0, 0.0], payload)

        # Semantically close query embedding (sim = 0.96)
        res, score, hit_type = cache.get(
            "Transformer attention explanation",
            query_embedding=[0.96, 0.28, 0.0],
        )
        assert res == payload
        assert hit_type == "semantic"
        assert score >= 0.90
        assert cache.stats.semantic_hits == 1

    def test_semantic_miss_below_threshold(self):
        cache = SemanticCache(similarity_threshold=0.95)
        payload = {"answer": "Some answer."}
        cache.put("Query A", [1.0, 0.0], payload)

        res, score, hit_type = cache.get("Query B", query_embedding=[0.5, 0.86])
        assert res is None
        assert hit_type == "miss"
        assert cache.stats.misses == 1

    def test_lru_eviction(self):
        cache = SemanticCache(max_size=2)
        cache.put("Q1", [1.0], {"a": 1})
        cache.put("Q2", [2.0], {"a": 2})
        assert cache.size() == 2

        # Add Q3 -> should evict Q1 (oldest accessed)
        cache.put("Q3", [3.0], {"a": 3})
        assert cache.size() == 2
        res, _, _ = cache.get("Q1")
        assert res is None
        res, _, _ = cache.get("Q2")
        assert res == {"a": 2}

    def test_ttl_expiry(self):
        cache = SemanticCache(default_ttl_seconds=0.01)
        cache.put("Ephemeral query", [1.0], {"a": 1})
        time.sleep(0.02)

        res, _, hit_type = cache.get("Ephemeral query")
        assert res is None
        assert hit_type == "miss"

    def test_cache_telemetry_stats(self):
        cache = SemanticCache()
        cache.put("Q1", [1.0], {"answer": "A1"})

        cache.get("Q1")  # Hit
        cache.get("Non-existent")  # Miss

        stats = cache.stats.to_dict()
        assert stats["total_lookups"] == 2
        assert stats["exact_hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
