"""
Hierarchical Semantic Cache for RAG Pipelines.
Provides two-tier caching:
1. Exact-match KV Cache (sub-millisecond normalized string lookup)
2. Semantic Vector Cache (cosine similarity lookup with intent & keyword verification)

Reduces pipeline latency from ~1-3s to <5ms and drastically reduces LLM token costs.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CachedItem:
    """A cached query response entry."""

    query: str
    query_norm: str
    embedding: list[float]
    payload: dict
    created_at: float
    ttl_seconds: float
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class CacheStats:
    """Telemetry and efficiency metrics for the semantic cache."""

    total_lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    saved_latency_ms: float = 0.0

    @property
    def total_hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        return round(self.total_hits / max(self.total_lookups, 1), 3)

    def to_dict(self) -> dict:
        return {
            "total_lookups": self.total_lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "saved_latency_ms": round(self.saved_latency_ms, 1),
        }


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two unit/normalized or standard vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def _normalize_query(query: str) -> str:
    """Canonical string for exact cache key matching."""
    s = query.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


class SemanticCache:
    """
    Two-tier hierarchical semantic cache with LRU eviction and TTL expiry.
    """

    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.95,
        default_ttl_seconds: float = 86400.0,  # 24 hours
        keyword_overlap_threshold: float = 0.60,
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.keyword_overlap_threshold = keyword_overlap_threshold

        self._exact_map: dict[str, CachedItem] = {}
        self._items: list[CachedItem] = []
        self.stats = CacheStats()

    def get(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        threshold: float | None = None,
    ) -> tuple[dict | None, float, str]:
        """
        Query the cache.

        Returns:
            (payload, similarity_score, hit_type) where hit_type is 'exact', 'semantic', or 'miss'.
        """
        self.stats.total_lookups += 1
        query_norm = _normalize_query(query)
        effective_threshold = threshold or self.similarity_threshold

        # 1. Tier 1: Exact string match
        if query_norm in self._exact_map:
            item = self._exact_map[query_norm]
            if not item.is_expired:
                item.hits += 1
                item.last_accessed = time.time()
                self.stats.exact_hits += 1
                saved_lat = item.payload.get("total_latency_ms", 1200.0)
                self.stats.saved_latency_ms += saved_lat
                logger.info(f"SemanticCache Tier 1 (Exact Hit) for: '{query[:40]}'")
                return item.payload, 1.0, "exact"
            else:
                self._evict_item(item)

        # 2. Tier 2: Semantic embedding match
        if query_embedding and self._items:
            q_tokens = set(re.findall(r"\w+", query_norm))
            best_score = 0.0
            best_item: CachedItem | None = None

            for item in self._items:
                if item.is_expired:
                    continue
                sim = _cosine_similarity(query_embedding, item.embedding)
                if sim > best_score:
                    best_score = sim
                    best_item = item

            if best_item and best_score >= effective_threshold:
                # Anti-Poisoning: Verify keyword overlap to prevent false positive semantic drift
                c_tokens = set(re.findall(r"\w+", best_item.query_norm))
                overlap = (
                    len(q_tokens & c_tokens) / max(min(len(q_tokens), len(c_tokens)), 1)
                    if q_tokens and c_tokens
                    else 1.0
                )

                if overlap >= self.keyword_overlap_threshold:
                    best_item.hits += 1
                    best_item.last_accessed = time.time()
                    self.stats.semantic_hits += 1
                    saved_lat = best_item.payload.get("total_latency_ms", 1200.0)
                    self.stats.saved_latency_ms += saved_lat
                    logger.info(
                        f"SemanticCache Tier 2 (Semantic Hit, sim={best_score:.3f}) for: '{query[:40]}'"
                    )
                    return best_item.payload, round(best_score, 4), "semantic"

        self.stats.misses += 1
        return None, 0.0, "miss"

    def put(
        self,
        query: str,
        query_embedding: list[float],
        payload: dict,
        ttl_seconds: float | None = None,
    ) -> None:
        """Insert or update a response in the cache."""
        if not query.strip() or not payload:
            return

        query_norm = _normalize_query(query)
        ttl = ttl_seconds or self.default_ttl_seconds

        # Enforce LRU eviction if full
        if len(self._items) >= self.max_size:
            self._evict_lru()

        item = CachedItem(
            query=query,
            query_norm=query_norm,
            embedding=query_embedding,
            payload=payload,
            created_at=time.time(),
            ttl_seconds=ttl,
        )

        self._exact_map[query_norm] = item
        self._items.append(item)

    def _evict_item(self, item: CachedItem) -> None:
        """Evict a specific item."""
        if item.query_norm in self._exact_map:
            del self._exact_map[item.query_norm]
        if item in self._items:
            self._items.remove(item)

    def _evict_lru(self) -> None:
        """Evict least recently accessed item."""
        if not self._items:
            return
        # Find oldest access
        self._items.sort(key=lambda x: x.last_accessed)
        oldest = self._items.pop(0)
        if oldest.query_norm in self._exact_map:
            del self._exact_map[oldest.query_norm]

    def clear(self) -> None:
        """Clear all cache items."""
        self._exact_map.clear()
        self._items.clear()

    def size(self) -> int:
        """Return number of cached items."""
        return len(self._items)
