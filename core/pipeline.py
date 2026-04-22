"""
Pipeline Orchestrator.
Wires all layers together: retrieval → generation → reliability → logging.
Single entry point: pipeline.run(query) → PipelineResult.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

from core.retriever import Retriever, RetrievedChunk
from core.generator import Generator, GeneratedAnswer, Citation
from core.reliability import ReliabilityChecker, ReliabilityReport
from core.query_handler import QueryHandler, ProcessedQuery

logger = logging.getLogger(__name__)


ABSTENTION_MESSAGE = (
    "⚠️ I don't have enough evidence to answer this question reliably. "
    "The retrieved documents don't provide sufficient support for a confident answer.\n\n"
    "**Suggestions:**\n"
    "- Try rephrasing your question to be more specific\n"
    "- Upload additional relevant documents\n"
    "- Check if your question is within the scope of the uploaded documents"
)


@dataclass
class PipelineResult:
    """Complete result from a single pipeline run."""
    # Input
    query: str
    processed_query: ProcessedQuery | None = None

    # Retrieval
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)

    # Generation
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    generated: GeneratedAnswer | None = None

    # Reliability
    reliability: ReliabilityReport | None = None
    should_abstain: bool = False
    abstention_message: str | None = None

    # Observability
    latency_ms: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0
    model: str = ""

    @property
    def display_answer(self) -> str:
        """The answer to show the user (abstention message if abstaining)."""
        if self.should_abstain:
            return self.abstention_message or ABSTENTION_MESSAGE
        return self.answer

    def to_dict(self) -> dict:
        """Serialize to dict for logging."""
        return {
            "query": self.query,
            "rewritten_query": self.processed_query.rewritten if self.processed_query else None,
            "was_rewritten": self.processed_query.was_rewritten if self.processed_query else False,
            "answer": self.answer,
            "display_answer": self.display_answer,
            "should_abstain": self.should_abstain,
            "abstention_message": self.abstention_message,
            "num_chunks_retrieved": len(self.retrieved_chunks),
            "retrieval_scores": [c.score for c in self.retrieved_chunks],
            "citations": [
                {"source_index": c.source_index, "chunk_id": c.chunk_id, "quote": c.quote}
                for c in self.citations
            ],
            "reliability": {
                "citation_score": self.reliability.citation_score if self.reliability else None,
                "grounding_score": self.reliability.grounding_score if self.reliability else None,
                "confidence": self.reliability.confidence if self.reliability else None,
                "unsupported_ratio": self.reliability.unsupported_ratio if self.reliability else None,
                "should_abstain": self.reliability.should_abstain if self.reliability else None,
                "abstention_reason": self.reliability.abstention_reason if self.reliability else None,
                "verdict": self.reliability.verdict if self.reliability else None,
                "details": self.reliability.details if self.reliability else None,
            },
            "latency_ms": self.latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "model": self.model,
        }


class Pipeline:
    """
    Main RAG pipeline orchestrator.
    query → query_rewrite → retrieval → reranking → generation → reliability checks → result
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        reliability_checker: ReliabilityChecker | None = None,
        query_handler: QueryHandler | None = None,
        query_logger=None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.reliability_checker = reliability_checker or ReliabilityChecker()
        self.query_handler = query_handler
        self.query_logger = query_logger

    def run(
        self,
        query: str,
        top_k: int | None = None,
        temperature: float = 0.3,
        enable_rewrite: bool = True,
        enable_reranking: bool = False,
    ) -> PipelineResult:
        """
        Execute the full RAG pipeline.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.
            temperature: LLM temperature.
            enable_rewrite: Whether to rewrite vague queries.
            enable_reranking: Whether to LLM-rerank retrieved chunks.

        Returns:
            PipelineResult with answer, citations, reliability, and latency.
        """
        result = PipelineResult(
            query=query,
            model=self.generator.provider.get_model_name(),
        )
        pipeline_start = time.time()

        # ── Layer 0: Query Processing ─────────────────────────────────────
        retrieval_query = query
        if self.query_handler and enable_rewrite:
            t0 = time.time()
            try:
                processed = self.query_handler.process(query, enable_rewrite=True)
                result.processed_query = processed
                retrieval_query = processed.effective_query
            except Exception as e:
                logger.warning(f"Query processing failed, using original: {e}")
            result.latency_ms["query_processing"] = round((time.time() - t0) * 1000, 1)

        # ── Layer 1: Retrieval ────────────────────────────────────────────
        t0 = time.time()
        try:
            chunks = self.retriever.retrieve(
                retrieval_query,
                top_k=top_k,
                enable_reranking=enable_reranking,
            )
            result.retrieved_chunks = chunks
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            chunks = []
        result.latency_ms["retrieval"] = round((time.time() - t0) * 1000, 1)

        # ── Layer 2: Generation ───────────────────────────────────────────
        t0 = time.time()
        try:
            generated = self.generator.generate(
                query=query,  # Use ORIGINAL query for generation (not rewritten)
                chunks=chunks,
                temperature=temperature,
            )
            result.generated = generated
            result.answer = generated.answer
            result.citations = generated.citations
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            generated = GeneratedAnswer(
                answer="An error occurred during generation.",
                self_confidence=0.0,
            )
            result.generated = generated
            result.answer = generated.answer
        result.latency_ms["generation"] = round((time.time() - t0) * 1000, 1)

        # ── Layer 3: Reliability Checks ───────────────────────────────────
        t0 = time.time()
        try:
            reliability = self.reliability_checker.check(
                answer=generated,
                chunks=chunks,
            )
            result.reliability = reliability
            result.should_abstain = reliability.should_abstain
            if reliability.should_abstain:
                result.abstention_message = (
                    f"{ABSTENTION_MESSAGE}\n\n"
                    f"**Reason:** {reliability.abstention_reason}"
                )
        except Exception as e:
            logger.error(f"Reliability check failed: {e}")
            result.reliability = ReliabilityReport(
                confidence=0.0,
                should_abstain=True,
                abstention_reason=f"Reliability check error: {str(e)}",
            )
            result.should_abstain = True
        result.latency_ms["reliability"] = round((time.time() - t0) * 1000, 1)

        # ── Total Latency ─────────────────────────────────────────────────
        result.total_latency_ms = round((time.time() - pipeline_start) * 1000, 1)

        # ── Logging ───────────────────────────────────────────────────────
        if self.query_logger:
            try:
                self.query_logger.log(result)
            except Exception as e:
                logger.error(f"Logging failed: {e}")

        logger.info(
            f"Pipeline complete: {result.reliability.verdict_emoji if result.reliability else '?'} "
            f"confidence={result.reliability.confidence if result.reliability else 'N/A'} "
            f"latency={result.total_latency_ms}ms"
        )

        return result
