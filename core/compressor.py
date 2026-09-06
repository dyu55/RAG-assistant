"""
Context Compression & Lost-in-the-Middle Mitigation Module.
Provides:
1. Position-Aware U-shaped Reordering (Lost-in-the-Middle mitigation)
2. Cross-Chunk Sentence Deduplication (Redundancy reduction)
3. Context Compaction & Token Optimization
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def _sentence_jaccard(s1: str, s2: str) -> float:
    """Calculate word-level Jaccard similarity between two sentences."""
    w1 = set(re.findall(r"\w+", s1.lower()))
    w2 = set(re.findall(r"\w+", s2.lower()))
    if not w1 or not w2:
        return 0.0
    intersection = len(w1 & w2)
    union = len(w1 | w2)
    return intersection / max(union, 1)


@dataclass
class CompressionStats:
    """Statistics for context compression."""

    original_chars: int = 0
    compressed_chars: int = 0
    sentences_dropped: int = 0
    chunks_reordered: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_chars == 0:
            return 1.0
        return round(self.compressed_chars / self.original_chars, 3)


class ContextCompressor:
    """
    Compresses and restructures retrieved chunks to optimize prompt attention budget.
    """

    def __init__(
        self,
        dedup_similarity_threshold: float = 0.80,
        min_sentence_length: int = 20,
    ):
        self.dedup_similarity_threshold = dedup_similarity_threshold
        self.min_sentence_length = min_sentence_length

    def reorder_lost_in_the_middle(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Reorder chunks to combat the 'Lost in the Middle' phenomenon.

        LLMs attend most strongly to the start and end of prompt context.
        This reorders sorted chunks (rank 0 = highest relevance) into a U-shaped layout:
        Rank 0 at start (index 0), Rank 1 at end (index -1), Rank 2 at index 1, Rank 3 at index -2...
        """
        if len(chunks) <= 2:
            return list(chunks)

        # Assumes incoming chunks are sorted by relevance descending
        reordered: list[RetrievedChunk | None] = [None] * len(chunks)
        left = 0
        right = len(chunks) - 1

        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                reordered[left] = chunk
                left += 1
            else:
                reordered[right] = chunk
                right -= 1

        return [c for c in reordered if c is not None]

    def deduplicate_sentences(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[RetrievedChunk], int]:
        """
        Remove redundant sentences across chunks while preserving chunk objects and order.
        Returns (compressed_chunks, dropped_sentence_count).
        """
        seen_sentences: list[str] = []
        dropped_count = 0
        compressed_chunks: list[RetrievedChunk] = []

        for chunk in chunks:
            # Split text into candidate sentences
            raw_sentences = re.split(r"(?<=[.!?。！？\n])\s+", chunk.text)
            kept_sentences = []

            for sent in raw_sentences:
                clean_sent = sent.strip()
                if not clean_sent:
                    continue

                if len(clean_sent) < self.min_sentence_length:
                    kept_sentences.append(sent)
                    continue

                # Check similarity against previously seen sentences
                is_duplicate = False
                for seen in seen_sentences:
                    if _sentence_jaccard(clean_sent, seen) >= self.dedup_similarity_threshold:
                        is_duplicate = True
                        dropped_count += 1
                        break

                if not is_duplicate:
                    seen_sentences.append(clean_sent)
                    kept_sentences.append(sent)

            # Reconstruct chunk with compressed text
            # Create a shallow copy or update chunk with filtered text
            new_text = " ".join(kept_sentences).strip()
            if not new_text:
                new_text = chunk.text  # Safety: retain original if all pruned

            # We create a new RetrievedChunk with updated text
            from core.retriever import RetrievedChunk

            compressed_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=new_text,
                score=chunk.score,
                metadata=dict(chunk.metadata),
                rerank_score=chunk.rerank_score,
            )
            compressed_chunks.append(compressed_chunk)

        return compressed_chunks, dropped_count

    def compress(
        self,
        chunks: list[RetrievedChunk],
        enable_reordering: bool = True,
        enable_dedup: bool = True,
    ) -> tuple[list[RetrievedChunk], CompressionStats]:
        """
        Run full context compression pipeline:
        1. Cross-chunk sentence deduplication
        2. Position-aware U-shaped reordering
        """
        if not chunks:
            return [], CompressionStats()

        orig_chars = sum(len(c.text) for c in chunks)
        active_chunks = list(chunks)
        dropped_sentences = 0

        if enable_dedup:
            active_chunks, dropped_sentences = self.deduplicate_sentences(active_chunks)

        if enable_reordering:
            active_chunks = self.reorder_lost_in_the_middle(active_chunks)

        comp_chars = sum(len(c.text) for c in active_chunks)
        stats = CompressionStats(
            original_chars=orig_chars,
            compressed_chars=comp_chars,
            sentences_dropped=dropped_sentences,
            chunks_reordered=len(active_chunks) if enable_reordering else 0,
        )

        return active_chunks, stats


default_compressor = ContextCompressor()
