"""
Retriever.
Performs vector similarity search against ChromaDB to find relevant document chunks.
Supports optional LLM-based reranking for improved precision.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from config import settings
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


RERANK_SYSTEM_PROMPT = """You are a relevance scoring system. Given a user question and a document chunk, score how relevant the chunk is to answering the question.

Score from 0.0 to 1.0:
- 1.0: Chunk directly and completely answers the question
- 0.7-0.9: Chunk contains highly relevant information
- 0.4-0.6: Chunk is somewhat relevant but not directly answering
- 0.1-0.3: Chunk has minimal relevance
- 0.0: Chunk is completely irrelevant

Respond with ONLY a JSON object:
{"relevance_score": 0.85, "reason": "brief explanation"}
"""


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with its relevance score."""
    chunk_id: str
    text: str
    score: float              # Similarity score (higher = more relevant)
    rerank_score: float = -1  # LLM reranking score (-1 = not reranked)
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("filename", "unknown")

    @property
    def effective_score(self) -> float:
        """Best available score (rerank if available, else similarity)."""
        return self.rerank_score if self.rerank_score >= 0 else self.score


class Retriever:
    """
    Retrieves relevant document chunks from ChromaDB via vector similarity search.
    Supports optional LLM-based reranking for improved precision.
    """

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str | None = None,
        rerank_provider=None,
    ):
        self.embedder = embedder
        self.collection = embedder.get_or_create_collection(collection_name)
        self.rerank_provider = rerank_provider  # OpenAI provider for LLM reranking

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        enable_reranking: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve.
            enable_reranking: If True, rerank results using LLM.

        Returns:
            List of RetrievedChunk objects, sorted by relevance (best first).
        """
        top_k = top_k or settings.TOP_K

        if self.collection.count() == 0:
            logger.warning("Collection is empty. No chunks to retrieve.")
            return []

        # If reranking, over-fetch to have a larger pool to rerank from
        fetch_k = top_k * 2 if enable_reranking else top_k
        fetch_k = min(fetch_k, self.collection.count())

        # Embed the query using the same embedding backend
        query_embedding = self.embedder.embed_query(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert ChromaDB results to RetrievedChunk objects
        # ChromaDB returns cosine distances; convert to similarity scores
        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # Cosine distance → similarity: sim = 1 - distance
                distance = results["distances"][0][i]
                score = 1.0 - distance

                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    score=round(score, 4),
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                chunks.append(chunk)

        # Sort by score descending
        chunks.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...' "
            f"(best score: {chunks[0].score if chunks else 'N/A'})"
        )

        # Optional: LLM-based reranking
        if enable_reranking and self.rerank_provider and chunks:
            chunks = self._rerank(query, chunks)
            # After reranking, take only top_k
            chunks = chunks[:top_k]
            logger.info(
                f"Reranked to {len(chunks)} chunks "
                f"(best rerank score: {chunks[0].rerank_score if chunks else 'N/A'})"
            )
        else:
            chunks = chunks[:top_k]

        return chunks

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Rerank chunks using LLM-based relevance scoring.
        Each chunk is scored independently for relevance to the query.
        """
        for chunk in chunks:
            try:
                prompt = (
                    f"USER QUESTION: {query}\n\n"
                    f"DOCUMENT CHUNK:\n{chunk.text[:800]}\n\n"
                    f"Score the relevance of this chunk to the question."
                )
                raw = self.rerank_provider.generate_json(
                    prompt=prompt,
                    system_prompt=RERANK_SYSTEM_PROMPT,
                    temperature=0.1,
                )
                score = float(raw.get("relevance_score", 0.0))
                chunk.rerank_score = round(max(0.0, min(1.0, score)), 4)
            except Exception as e:
                logger.warning(f"Reranking failed for chunk {chunk.chunk_id[:8]}: {e}")
                chunk.rerank_score = chunk.score  # Fallback to original score

        # Sort by rerank score descending
        chunks.sort(key=lambda c: c.rerank_score, reverse=True)
        return chunks

    def has_documents(self) -> bool:
        """Check if the collection has any documents."""
        return self.collection.count() > 0

