"""
Unit tests for core/reranker.py (FastCrossEncoderReranker).
"""

from __future__ import annotations

from core.reranker import FastCrossEncoderReranker
from core.retriever import RetrievedChunk


class TestFastCrossEncoderReranker:
    def _make_chunk(self, chunk_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=score,
            metadata={},
        )

    def test_empty_chunks_returns_empty_list(self):
        reranker = FastCrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    def test_empty_query_returns_original_chunks(self):
        reranker = FastCrossEncoderReranker()
        chunks = [self._make_chunk("c1", "Sample text", 0.8)]
        assert reranker.rerank("   ", chunks) == chunks

    def test_exact_phrase_match_boosts_ranking(self):
        reranker = FastCrossEncoderReranker()
        # c1 has high initial embedding score but no exact match
        c1 = self._make_chunk(
            "c1", "Retrieval augmented systems are useful for machine learning.", 0.85
        )
        # c2 has lower initial score but contains exact query phrase
        c2 = self._make_chunk(
            "c2", "We introduce Corrective RAG (CRAG) for self-healing retrieval.", 0.60
        )

        reranked = reranker.rerank("Corrective RAG", [c1, c2])
        assert len(reranked) == 2
        # c2 should be boosted to rank 1 due to exact phrase match
        assert reranked[0].chunk_id == "c2"
        assert reranked[0].rerank_score > reranked[1].rerank_score

    def test_keyword_proximity_boosts_relevant_chunk(self):
        reranker = FastCrossEncoderReranker()
        # c1 has query terms scattered far apart
        c1 = self._make_chunk(
            "c1", "Transformer is powerful." + " padding text " * 40 + "Attention is useful.", 0.70
        )
        # c2 has query terms adjacent
        c2 = self._make_chunk(
            "c2", "The Transformer attention mechanism enables parallel token processing.", 0.70
        )

        reranked = reranker.rerank("Transformer attention", [c1, c2])
        assert reranked[0].chunk_id == "c2"

    def test_contextual_document_tag_boost(self):
        reranker = FastCrossEncoderReranker()
        c1 = self._make_chunk(
            "c1",
            "[Document: GraphRAG Architecture]\n\nTraversing entities and relationships.",
            0.65,
        )
        c2 = self._make_chunk("c2", "General knowledge graph traversal methods.", 0.65)

        reranked = reranker.rerank("GraphRAG architecture", [c2, c1])
        assert reranked[0].chunk_id == "c1"

    def test_top_n_truncation(self):
        reranker = FastCrossEncoderReranker()
        chunks = [self._make_chunk(f"c{i}", f"Chunk text {i}", 0.5) for i in range(10)]
        reranked = reranker.rerank("Chunk", chunks, top_n=3)
        assert len(reranked) == 3

    def test_rerank_scores_bounded_between_zero_and_one(self):
        reranker = FastCrossEncoderReranker()
        chunks = [
            self._make_chunk("c1", "Exact query match here.", 0.99),
            self._make_chunk("c2", "Unrelated noise.", 0.01),
        ]
        reranked = reranker.rerank("Exact query match", chunks)
        for c in reranked:
            assert 0.0 <= c.rerank_score <= 1.0
