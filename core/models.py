"""Shared retrieval and answer data models, independent of storage and providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A chunk retrieved from a vector store or graph traversal with its relevance score."""

    chunk_id: str
    text: str
    score: float  # Similarity score (higher = more relevant)
    rerank_score: float = -1  # LLM reranking score (-1 = not reranked)
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Filename of the originating document (vector path) or graph label."""
        # Prefer an explicit filename stored by the embedder. Graph/community
        # chunks typically don't have one, so we fall back to a label.
        return self.metadata.get("filename") or self.metadata.get("source", "vector")

    @property
    def retrieval_source(self) -> str:
        """Which retrieval path produced this chunk.

        One of:
        - "vector"     — ChromaDB cosine similarity (the default)
        - "graph"      — entity-anchored subgraph traversal (GraphRAG local)
        - "community"  — map-reduce over community summaries (GraphRAG global)
        """
        src = self.metadata.get("source")
        if src in {"vector", "graph", "community"}:
            return src
        return "vector"

    @property
    def effective_score(self) -> float:
        """Best available score (rerank if available, else similarity)."""
        return self.rerank_score if self.rerank_score >= 0 else self.score


@dataclass
class Citation:
    """A single citation linking an answer claim to a source chunk."""

    source_index: int
    chunk_id: str
    quote: str
    # Origin of the cited chunk: "vector", "graph", or "community".
    # Populated automatically from the RetrievedChunk.metadata.
    source_type: str = "vector"


@dataclass
class GeneratedAnswer:
    """Structured answer from the LLM with citations and confidence."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    self_confidence: float = 0.0
    reasoning: str = ""
    raw_response: dict = field(default_factory=dict)

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0
