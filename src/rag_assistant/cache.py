from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from .models import Answer, Question
from .text import normalize


@dataclass
class CacheEntry:
    query_text: str
    normalized_query: str
    vector: list[float] | None
    revision: int
    document_ids: tuple[str, ...]
    mode: str
    provider: str
    model: str
    answer: Answer
    timestamp: float


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class SemanticCache:
    """Thread-safe semantic cache supporting exact normalized query matching and

    vector cosine similarity matches scoped to document index revisions.
    """

    def __init__(
        self,
        ttl: float = 300.0,
        threshold: float = 0.92,
        max_entries: int = 500,
        enabled: bool = True,
    ):
        self.ttl = ttl
        self.threshold = threshold
        self.max_entries = max_entries
        self.enabled = enabled
        self._entries: list[CacheEntry] = []
        self._lock = threading.RLock()
        self.exact_hits: int = 0
        self.semantic_hits: int = 0
        self.misses: int = 0

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        return self.ttl > 0 and (now - entry.timestamp) >= self.ttl

    def get_exact(
        self, question: Question, revision: int, provider: str, model: str
    ) -> Answer | None:
        """Fast-path exact query cache lookup using normalized string matching."""
        if not self.enabled:
            return None

        now = time.monotonic()
        norm_q = normalize(question.text).lower()
        doc_ids = tuple(sorted(question.document_ids))

        with self._lock:
            # Clean expired entries
            self._entries = [e for e in self._entries if not self._is_expired(e, now)]

            for i in range(len(self._entries) - 1, -1, -1):
                entry = self._entries[i]
                if (
                    entry.revision == revision
                    and entry.provider == provider
                    and entry.model == model
                    and entry.document_ids == doc_ids
                    and entry.mode == question.mode
                    and entry.normalized_query == norm_q
                ):
                    self.exact_hits += 1
                    # LRU update
                    hit_entry = self._entries.pop(i)
                    self._entries.append(hit_entry)
                    ans = hit_entry.answer.model_copy(deep=True)
                    ans.id = uuid.uuid4().hex
                    ans.cached = True
                    return ans

            self.misses += 1
            return None

    def get_semantic(
        self,
        question: Question,
        revision: int,
        provider: str,
        model: str,
        vector: list[float],
        threshold: float | None = None,
    ) -> Answer | None:
        """Cosine similarity semantic lookup over cached question vectors."""
        if not self.enabled or not vector:
            return None

        now = time.monotonic()
        doc_ids = tuple(sorted(question.document_ids))
        effective_threshold = threshold if threshold is not None else self.threshold

        with self._lock:
            self._entries = [e for e in self._entries if not self._is_expired(e, now)]

            best_sim = -1.0
            best_idx = -1

            for i in range(len(self._entries) - 1, -1, -1):
                entry = self._entries[i]
                if (
                    entry.revision == revision
                    and entry.provider == provider
                    and entry.model == model
                    and entry.document_ids == doc_ids
                    and entry.mode == question.mode
                    and entry.vector is not None
                ):
                    sim = cosine_similarity(vector, entry.vector)
                    if sim >= effective_threshold and sim > best_sim:
                        best_sim = sim
                        best_idx = i

            if best_idx >= 0:
                self.semantic_hits += 1
                hit_entry = self._entries.pop(best_idx)
                self._entries.append(hit_entry)
                ans = hit_entry.answer.model_copy(deep=True)
                ans.id = uuid.uuid4().hex
                ans.cached = True
                ans.warnings = [
                    *ans.warnings,
                    f"Answer served from semantic cache (similarity: {best_sim:.3f})",
                ]
                return ans

            return None

    def put(
        self,
        question: Question,
        revision: int,
        provider: str,
        model: str,
        answer: Answer,
        vector: list[float] | None = None,
    ) -> None:
        """Record an answer in the semantic cache."""
        if not self.enabled:
            return

        now = time.monotonic()
        entry = CacheEntry(
            query_text=question.text,
            normalized_query=normalize(question.text).lower(),
            vector=vector,
            revision=revision,
            document_ids=tuple(sorted(question.document_ids)),
            mode=question.mode,
            provider=provider,
            model=model,
            answer=answer.model_copy(deep=True),
            timestamp=now,
        )

        with self._lock:
            self._entries.append(entry)
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        """Return cache performance statistics."""
        with self._lock:
            total_hits = self.exact_hits + self.semantic_hits
            total_lookups = total_hits + self.misses
            hit_rate = round(total_hits / max(1, total_lookups), 4)
            return {
                "enabled": self.enabled,
                "size": len(self._entries),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl,
                "threshold": self.threshold,
                "exact_hits": self.exact_hits,
                "semantic_hits": self.semantic_hits,
                "total_hits": total_hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }
