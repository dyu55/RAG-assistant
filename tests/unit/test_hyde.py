"""
Unit tests for core/hyde.py (HyDE Query Expansion).
"""

from __future__ import annotations

from unittest.mock import Mock

from core.hyde import HyDEGenerator, generate_hyde_passage


class TestHyDEGenerator:
    def test_empty_query_returns_empty_string(self):
        generator = HyDEGenerator()
        assert generator.generate_hypothetical_document("") == ""
        assert generator.generate_hypothetical_document("   ") == ""

    def test_heuristic_fallback_when_no_provider(self):
        generator = HyDEGenerator(provider=None)
        query = "How does vector indexing work in ChromaDB?"
        doc = generator.generate_hypothetical_document(query)

        assert len(doc) > 0
        assert "ChromaDB" in doc
        assert "indexing" in doc

    def test_llm_provider_generation_success(self):
        mock_provider = Mock()
        mock_provider.generate.return_value = (
            "Vector indexing in ChromaDB uses HNSW graphs for approximate nearest neighbor search."
        )
        generator = HyDEGenerator(provider=mock_provider)

        doc = generator.generate_hypothetical_document("How does vector indexing work in ChromaDB?")
        assert (
            doc
            == "Vector indexing in ChromaDB uses HNSW graphs for approximate nearest neighbor search."
        )
        assert mock_provider.generate.called

    def test_llm_provider_failure_falls_back_to_heuristic(self):
        mock_provider = Mock()
        mock_provider.generate.side_effect = RuntimeError("API timeout")
        generator = HyDEGenerator(provider=mock_provider)

        doc = generator.generate_hypothetical_document("Transformer attention mechanism")
        assert len(doc) > 0
        assert "Transformer" in doc

    def test_generate_hyde_passage_helper(self):
        doc = generate_hyde_passage("Graph neural networks")
        assert len(doc) > 0
        assert "Graph" in doc
