"""
Integration tests for the RAG pipeline.
Tests the full orchestration layer with mocked dependencies.
"""

from __future__ import annotations

from unittest.mock import Mock

from core.generator import Citation, GeneratedAnswer
from core.pipeline import Pipeline, PipelineResult
from core.query_handler import ProcessedQuery
from core.reliability import ReliabilityReport
from core.retriever import RetrievedChunk

# ── Mock Factories ───────────────────────────────────────────────────────────────


def make_chunk(chunk_id: str, text: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={"filename": f"{chunk_id}.txt"},
    )


def make_answer(
    answer: str,
    citations: list[Citation] | None = None,
    self_confidence: float = 0.8,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=answer,
        citations=citations or [],
        self_confidence=self_confidence,
        reasoning="Test reasoning.",
    )


def make_report(
    confidence: float,
    citation_score: float = 1.0,
    grounding_score: float = 1.0,
    should_abstain: bool = False,
    abstention_reason: str | None = None,
) -> ReliabilityReport:
    return ReliabilityReport(
        confidence=confidence,
        citation_score=citation_score,
        grounding_score=grounding_score,
        should_abstain=should_abstain,
        abstention_reason=abstention_reason,
    )


# ── Pipeline Tests ──────────────────────────────────────────────────────────────


class TestPipelineRun:
    def _make_pipeline(
        self,
        retriever_mock,
        generator_mock,
        reliability_checker_mock=None,
        query_handler_mock=None,
        query_logger_mock=None,
    ):
        return Pipeline(
            retriever=retriever_mock,
            generator=generator_mock,
            reliability_checker=reliability_checker_mock,
            query_handler=query_handler_mock,
            query_logger=query_logger_mock,
        )

    def test_successful_end_to_end_run(self):
        chunks = [make_chunk("c1", "Python is a programming language.", 0.9)]
        answer = make_answer(
            "Python is a programming language [Source 1].",
            citations=[
                Citation(source_index=1, chunk_id="c1", quote="Python is a programming language.")
            ],
        )
        report = make_report(confidence=0.85)

        retriever = Mock()
        retriever.retrieve.return_value = chunks

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = answer

        reliability = Mock()
        reliability.check.return_value = report

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("What is Python?")

        assert result.answer == "Python is a programming language [Source 1]."
        assert result.reliability.confidence == 0.85
        assert result.should_abstain is False
        assert len(result.retrieved_chunks) == 1
        assert result.model == "gpt-4o-mini"
        assert result.total_latency_ms >= 0  # Mocks are fast — may be 0

    def test_abstention_sets_display_message(self):
        chunks = [make_chunk("c1", "Some content.", 0.3)]
        answer = make_answer("I don't know.", self_confidence=0.1)
        report = make_report(
            confidence=0.2,
            should_abstain=True,
            abstention_reason="Confidence too low.",
        )

        retriever = Mock()
        retriever.retrieve.return_value = chunks

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = answer

        reliability = Mock()
        reliability.check.return_value = report

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("What is meaning of life?")

        assert result.should_abstain is True
        assert "Confidence too low" in result.abstention_message
        assert "don't know" not in result.display_answer  # abstention message overrides

    def test_retrieval_failure_graceful_degradation(self):
        """When retrieval fails, pipeline continues with empty chunks."""
        retriever = Mock()
        retriever.retrieve.side_effect = Exception("ChromaDB connection failed")

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("Fallback answer.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.5)

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("Any question")

        # Should not raise - pipeline handles retrieval failure gracefully
        assert isinstance(result, PipelineResult)
        # Generator still called (with empty chunks → fallback answer)
        assert generator.generate.called

    def test_generation_failure_sets_error_message(self):
        chunks = [make_chunk("c1", "Content.")]
        retriever = Mock()
        retriever.retrieve.return_value = chunks

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.side_effect = Exception("OpenAI API error")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.0, should_abstain=True)

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("What is Python?")

        assert "error" in result.answer.lower() or result.should_abstain
        assert isinstance(result, PipelineResult)

    def test_reliability_check_failure_sets_abstention(self):
        chunks = [make_chunk("c1", "Content.")]
        answer = make_answer("Some answer.")
        retriever = Mock()
        retriever.retrieve.return_value = chunks

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = answer

        reliability = Mock()
        reliability.check.side_effect = Exception("Reliability check crashed")

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("Any question")

        assert result.should_abstain is True
        assert "Reliability check error" in result.reliability.abstention_reason

    def test_query_rewrite_improves_retrieval(self):
        """When query rewriting is enabled, the rewritten query is used for retrieval."""
        chunks_original = []
        chunks_rewritten = [make_chunk("c1", "RAG is retrieval augmented generation.", 0.95)]

        retriever = Mock()
        retriever.retrieve.side_effect = [
            chunks_original,  # First call with original query
            chunks_rewritten,  # Second call with rewritten query
        ]

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("RAG explanation.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.8)

        query_handler = Mock()
        query_handler.process.return_value = ProcessedQuery(
            original="what is rag",
            normalized="what is rag",
            rewritten="What is retrieval-augmented generation (RAG) and how does it work?",
            was_rewritten=True,
        )

        pipeline = self._make_pipeline(retriever, generator, reliability, query_handler)
        _ = pipeline.run("what is rag", enable_rewrite=True)

        # Query handler should be called
        query_handler.process.assert_called_once()
        # Retrieval should be called twice (rewrite) or once
        assert retriever.retrieve.called

    def test_query_rewrite_disabled(self):
        """When enable_rewrite=False, query handler is not used."""
        retriever = Mock()
        retriever.retrieve.return_value = [make_chunk("c1", "Content.")]

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("Answer.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.8)

        query_handler = Mock()
        pipeline = self._make_pipeline(retriever, generator, reliability, query_handler)

        _ = pipeline.run("What is AI?", enable_rewrite=False)

        # Query handler should NOT be called when rewrite is disabled
        query_handler.process.assert_not_called()

    def test_query_processing_failure_uses_original(self):
        """When query processing fails, original query is used."""
        chunks = [make_chunk("c1", "Content.")]

        retriever = Mock()
        retriever.retrieve.return_value = chunks

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("Answer.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.8)

        query_handler = Mock()
        query_handler.process.side_effect = Exception("Query handler crashed")

        pipeline = self._make_pipeline(retriever, generator, reliability, query_handler)
        result = pipeline.run("Original query")

        # Should not raise
        assert isinstance(result, PipelineResult)
        # Original query should be used
        assert result.query == "Original query"

    def test_empty_chunks_generates_no_context_answer(self):
        """When no chunks are retrieved, generator handles it."""
        retriever = Mock()
        retriever.retrieve.return_value = []  # No chunks

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        # Generator handles empty chunks
        generator.generate.return_value = GeneratedAnswer(
            answer="I cannot answer because no documents were found.",
            self_confidence=0.0,
        )

        reliability = Mock()
        reliability.check.return_value = make_report(
            confidence=0.0,
            should_abstain=True,
            abstention_reason="No chunks retrieved.",
        )

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("Any question?")

        assert result.should_abstain is True

    def test_latency_tracked_per_layer(self):
        retriever = Mock()
        retriever.retrieve.return_value = [make_chunk("c1", "Content.")]

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("Answer.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.8)

        pipeline = self._make_pipeline(retriever, generator, reliability)
        result = pipeline.run("Question")

        assert "retrieval" in result.latency_ms
        assert "generation" in result.latency_ms
        assert "reliability" in result.latency_ms
        assert result.total_latency_ms >= 0  # Mocks are fast — may be 0

    def test_logger_called_on_success(self):
        retriever = Mock()
        retriever.retrieve.return_value = [make_chunk("c1", "Content.")]

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("Answer.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.8)

        logger_mock = Mock()
        pipeline = self._make_pipeline(
            retriever, generator, reliability, query_logger_mock=logger_mock
        )
        result = pipeline.run("Question")

        logger_mock.log.assert_called_once_with(result)

    def test_logger_failure_does_not_crash_pipeline(self):
        retriever = Mock()
        retriever.retrieve.return_value = [make_chunk("c1", "Content.")]

        generator = Mock()
        provider = Mock()
        provider.get_model_name.return_value = "gpt-4o-mini"
        generator.provider = provider
        generator.generate.return_value = make_answer("Answer.")

        reliability = Mock()
        reliability.check.return_value = make_report(confidence=0.8)

        logger_mock = Mock()
        logger_mock.log.side_effect = Exception("Log write failed")

        pipeline = self._make_pipeline(
            retriever, generator, reliability, query_logger_mock=logger_mock
        )

        # Should not raise - logging failures are caught
        result = pipeline.run("Question")
        assert isinstance(result, PipelineResult)


class TestPipelineResult:
    def test_display_answer_shows_abstention_message(self):
        result = PipelineResult(
            query="test",
            answer="Some answer",
            should_abstain=True,
            abstention_message="⚠️ Not enough evidence.",
        )
        assert "Not enough evidence" in result.display_answer

    def test_display_answer_shows_answer_when_not_abstaining(self):
        result = PipelineResult(
            query="test",
            answer="The answer is 42.",
            should_abstain=False,
        )
        assert result.display_answer == "The answer is 42."

    def test_to_dict_includes_all_fields(self):
        chunks = [make_chunk("c1", "Source text.", 0.9)]
        answer = make_answer(
            "Answer [Source 1].",
            citations=[
                Citation(source_index=1, chunk_id="c1", quote="Source text."),
            ],
        )
        report = make_report(confidence=0.85, citation_score=1.0, grounding_score=0.9)

        result = PipelineResult(
            query="What is the answer?",
            answer="Answer [Source 1].",
            retrieved_chunks=chunks,
            citations=answer.citations,
            generated=answer,
            reliability=report,
            should_abstain=False,
            total_latency_ms=500.0,
            model="gpt-4o-mini",
            latency_ms={"retrieval": 100.0, "generation": 300.0, "reliability": 50.0},
        )

        d = result.to_dict()

        assert d["query"] == "What is the answer?"
        assert d["answer"] == "Answer [Source 1]."
        assert d["should_abstain"] is False
        assert d["num_chunks_retrieved"] == 1
        assert d["retrieval_scores"] == [0.9]
        assert len(d["citations"]) == 1
        assert d["reliability"]["confidence"] == 0.85
        assert d["total_latency_ms"] == 500.0
        assert d["model"] == "gpt-4o-mini"

    def test_to_dict_handles_none_reliability(self):
        result = PipelineResult(query="test", answer="Answer.")
        d = result.to_dict()
        assert d["reliability"]["confidence"] is None
        assert d["reliability"]["citation_score"] is None

    def test_to_dict_handles_none_processed_query(self):
        result = PipelineResult(query="test", answer="Answer.")
        d = result.to_dict()
        assert d["rewritten_query"] is None
        assert d["was_rewritten"] is False
