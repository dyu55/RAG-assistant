"""Unit tests for Corrective RAG (CRAG) retrieval evaluation and query decomposition."""

from __future__ import annotations

from core.crag import CRAGAction, CRAGEvaluator
from core.retriever import RetrievedChunk


class TestCRAGEvaluator:
    def _make_chunk(self, text: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="chunk-1",
            text=text,
            score=score,
            metadata={"source": "test"},
        )

    def test_evaluate_empty_chunks_returns_incorrect(self):
        evaluator = CRAGEvaluator()
        res = evaluator.evaluate("What is RAG?", [])
        assert res.is_incorrect
        assert res.action == CRAGAction.INCORRECT
        assert res.confidence == 0.0

    def test_evaluate_high_score_returns_correct(self):
        evaluator = CRAGEvaluator(correct_threshold=0.65)
        c1 = self._make_chunk("Retrieval-Augmented Generation (RAG) is an AI framework.", 0.85)
        c2 = self._make_chunk("RAG combines vector search with LLMs.", 0.75)

        res = evaluator.evaluate("What is RAG?", [c1, c2])
        assert res.is_correct
        assert res.action == CRAGAction.CORRECT
        assert res.confidence >= 0.65

    def test_evaluate_medium_score_returns_ambiguous_with_suggestions(self):
        evaluator = CRAGEvaluator(correct_threshold=0.65, incorrect_threshold=0.35)
        # Moderate score with partial keyword relevance
        c1 = self._make_chunk("Some partial information about Transformer architectures.", 0.50)

        res = evaluator.evaluate("How does Transformer attention and LSTM compare?", [c1])
        assert res.is_ambiguous
        assert res.action == CRAGAction.AMBIGUOUS
        assert len(res.suggested_queries) > 0

    def test_evaluate_low_score_returns_incorrect(self):
        evaluator = CRAGEvaluator(correct_threshold=0.65, incorrect_threshold=0.35)
        c1 = self._make_chunk("Unrelated recipe for apple pie.", 0.15)

        res = evaluator.evaluate("What is quantum computing?", [c1])
        assert res.is_incorrect
        assert res.action == CRAGAction.INCORRECT

    def test_decompose_and_expand_query(self):
        evaluator = CRAGEvaluator()
        # Comparison with 'and'
        exp = evaluator.decompose_and_expand_query(
            "Explain Transformer attention and LSTM recurrent cells"
        )
        assert len(exp) > 0
        assert any("Transformer attention" in q for q in exp)

        # Question word stripping
        exp2 = evaluator.decompose_and_expand_query("What is Kubernetes container orchestration?")
        assert any("Kubernetes container orchestration" in q for q in exp2)
