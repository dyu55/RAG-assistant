"""
Unit tests for core/query_handler.py
Tests the query preprocessing pipeline with mocked LLM.
"""

from __future__ import annotations

from unittest.mock import Mock

from core.query_handler import ProcessedQuery, QueryHandler


class TestQueryHandlerNormalize:
    handler = QueryHandler(provider=None)  # No LLM needed for normalization

    def test_strips_whitespace(self):
        result = self.handler._normalize("  hello   world  ")
        assert result == "hello world"

    def test_collapse_multiple_spaces(self):
        result = self.handler._normalize("hello    world\n\n  test")
        assert "    " not in result
        assert "  " not in result

    def test_removes_excessive_punctuation(self):
        result = self.handler._normalize("What is this????")
        assert "?" in result
        assert result.count("?") == 1

    def test_removes_surrounding_quotes(self):
        result = self.handler._normalize('"hello world"')
        assert result == "hello world"

        result2 = self.handler._normalize("'hello world'")
        assert result2 == "hello world"

    def test_no_change_for_normal_query(self):
        result = self.handler._normalize("What is retrieval-augmented generation?")
        assert "?" in result
        assert "retrieval-augmented" in result


class TestQueryHandlerShouldRewrite:
    handler = QueryHandler(provider=None)

    def test_short_queries_should_rewrite(self):
        assert self.handler._should_rewrite("what is rag") is True
        assert self.handler._should_rewrite("how to test") is True

    def test_vague_phrases_should_rewrite(self):
        assert self.handler._should_rewrite("tell me about python") is True
        assert self.handler._should_rewrite("explain authentication") is True
        assert self.handler._should_rewrite("help me understand this") is True

    def test_short_specific_queries_should_rewrite(self):
        # Short queries (<=4 words) always trigger rewrite regardless of specificity
        assert self.handler._should_rewrite("what is rag") is True
        assert self.handler._should_rewrite("python history") is True

    def test_long_vague_queries_should_rewrite(self):
        # Long queries starting with vague phrases trigger rewrite
        assert self.handler._should_rewrite("what is the meaning of python in programming") is True
        assert self.handler._should_rewrite("describe how authentication works in detail") is True


class TestQueryHandlerProcess:
    def test_full_pipeline_with_normalization(self):
        handler = QueryHandler(provider=None)
        result = handler.process("  what   is  rag?  ")

        assert isinstance(result, ProcessedQuery)
        assert result.normalized == "what is rag?"
        assert result.was_rewritten is False
        assert result.effective_query == result.normalized

    def test_full_pipeline_with_rewrite(self):
        """When provider is available and query is vague, rewrite is attempted."""
        mock_provider = Mock()
        # Return a distinctly different query
        mock_provider.generate.return_value = (
            "Retrieval-Augmented Generation (RAG) is a technique that combines..."
        )

        handler = QueryHandler(provider=mock_provider)
        result = handler.process("what is rag", enable_rewrite=True)

        assert isinstance(result, ProcessedQuery)
        # Should have called LLM
        mock_provider.generate.assert_called()
        # Should be rewritten (clearly different from normalized "what is rag")
        assert result.was_rewritten is True
        assert len(result.rewritten) > len("what is rag")
        assert result.effective_query == result.rewritten

    def test_rewrite_disabled_skips_llm(self):
        mock_provider = Mock()
        handler = QueryHandler(provider=mock_provider)

        result = handler.process("what is rag", enable_rewrite=False)

        mock_provider.generate.assert_not_called()
        assert result.was_rewritten is False

    def test_no_provider_uses_normalized_query(self):
        handler = QueryHandler(provider=None)
        result = handler.process("what is rag")

        assert result.was_rewritten is False
        assert result.effective_query == result.normalized

    def test_llm_failure_falls_back_to_normalized(self):
        mock_provider = Mock()
        mock_provider.generate.side_effect = Exception("API error")

        handler = QueryHandler(provider=mock_provider)
        result = handler.process("what is rag", enable_rewrite=True)

        # Should not raise - falls back gracefully
        assert isinstance(result, ProcessedQuery)
        assert result.was_rewritten is False
        assert result.effective_query == result.normalized

    def test_identical_rewrite_considered_not_rewritten(self):
        """If LLM returns the same query, it should not be marked as rewritten."""
        mock_provider = Mock()
        # LLM returns the query unchanged
        mock_provider.generate.return_value = "What is Python?"  # Same as input after normalization

        handler = QueryHandler(provider=mock_provider)
        result = handler.process("What is Python?", enable_rewrite=True)

        assert result.was_rewritten is False

    def test_effective_query_property(self):
        handler = QueryHandler(provider=None)
        result = handler.process("hello")
        assert result.effective_query == result.normalized


class TestProcessedQuery:
    def test_effective_query_uses_rewritten_when_available(self):
        pq = ProcessedQuery(
            original="hi",
            normalized="hi",
            rewritten="Hello, how can I help you?",
            was_rewritten=True,
        )
        assert pq.effective_query == pq.rewritten

    def test_effective_query_uses_normalized_when_not_rewritten(self):
        pq = ProcessedQuery(
            original="hi",
            normalized="hi",
            rewritten="Hello",
            was_rewritten=False,
        )
        assert pq.effective_query == pq.normalized
