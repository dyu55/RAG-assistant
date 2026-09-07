"""
Hierarchical Semantic Cache for RAG Pipelines.
Provides two-tier caching:
1. Exact-match KV Cache (sub-millisecond normalized string lookup)
2. Semantic Vector Cache (cosine similarity lookup with intent & keyword verification)

Caches verified responses with isolated payloads and bounded storage.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock

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
    namespace: str = ""
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) >= self.ttl_seconds


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
    return re.sub(r"\s+", " ", query.casefold().strip())


class SemanticCache:
    """Thread-safe LRU cache with TTL and optional semantic lookup.

    Namespaces isolate pipelines and run options. Exact normalization preserves
    punctuation so queries about C++ and C# cannot share an exact entry.
    """

    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.95,
        default_ttl_seconds: float = 86400.0,
        keyword_overlap_threshold: float = 0.60,
    ):
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self.keyword_overlap_threshold = keyword_overlap_threshold
        self._entries: OrderedDict[tuple[str, str], CachedItem] = OrderedDict()
        self._lock = RLock()
        self.stats = CacheStats()

    def _prune_expired(self):
        for key in [key for key, item in self._entries.items() if item.is_expired]:
            del self._entries[key]

    def _hit(self, key, item, similarity, hit_type):
        item.hits += 1
        item.last_accessed = time.time()
        self._entries.move_to_end(key)
        if hit_type == "exact":
            self.stats.exact_hits += 1
        else:
            self.stats.semantic_hits += 1
        self.stats.saved_latency_ms += item.payload.get("total_latency_ms", 0.0)
        return deepcopy(item.payload), round(similarity, 4), hit_type

    def get(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        threshold: float | None = None,
        *,
        namespace: str = "",
    ) -> tuple[dict | None, float, str]:
        """Return an independent payload, similarity and exact/semantic/miss status."""
        with self._lock:
            self.stats.total_lookups += 1
            self._prune_expired()
            query_norm = _normalize_query(query)
            key = (namespace, query_norm)
            item = self._entries.get(key)
            if item is not None:
                return self._hit(key, item, 1.0, "exact")

            effective_threshold = self.similarity_threshold if threshold is None else threshold
            best = None
            best_score = -1.0
            if query_embedding:
                q_tokens = set(re.findall(r"\w+", query_norm))
                for candidate_key, candidate in self._entries.items():
                    if candidate.namespace != namespace or not candidate.embedding:
                        continue
                    c_tokens = set(re.findall(r"\w+", candidate.query_norm))
                    overlap = len(q_tokens & c_tokens) / max(min(len(q_tokens), len(c_tokens)), 1)
                    if overlap < self.keyword_overlap_threshold:
                        continue
                    similarity = _cosine_similarity(query_embedding, candidate.embedding)
                    if similarity >= effective_threshold and similarity > best_score:
                        best, best_score = (candidate_key, candidate), similarity
            if best is not None:
                return self._hit(*best, best_score, "semantic")
            self.stats.misses += 1
            return None, 0.0, "miss"

    def put(
        self,
        query: str,
        query_embedding: list[float],
        payload: dict,
        ttl_seconds: float | None = None,
        *,
        namespace: str = "",
    ) -> None:
        """Upsert one entry without evicting unrelated entries on replacement."""
        if not query.strip() or not payload:
            return
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        key = (namespace, _normalize_query(query))
        with self._lock:
            self._prune_expired()
            self._entries.pop(key, None)
            if ttl <= 0:
                return
            self._entries[key] = CachedItem(
                query=query,
                query_norm=key[1],
                namespace=namespace,
                embedding=list(query_embedding),
                payload=deepcopy(payload),
                created_at=time.time(),
                ttl_seconds=ttl,
            )
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Clear entries while retaining cumulative telemetry."""
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        """Return the number of live entries."""
        with self._lock:
            self._prune_expired()
            return len(self._entries)
