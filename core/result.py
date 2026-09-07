"""Pipeline output and log serialization, shared by orchestration and UI."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.models import Citation, GeneratedAnswer, RetrievedChunk
from core.query_handler import ProcessedQuery
from core.reliability import ReliabilityReport

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

    # Routing & observability
    route_mode: str = "vector-only"
    route_confidence: float = 0.0
    route_reason: str = ""
    latency_ms: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0
    model: str = ""
    metadata: dict = field(default_factory=dict)

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
            "route_mode": self.route_mode,
            "route_confidence": self.route_confidence,
            "route_reason": self.route_reason,
            "num_chunks_retrieved": len(self.retrieved_chunks),
            "retrieval_scores": [c.score for c in self.retrieved_chunks],
            "sources_used": [c.retrieval_source for c in self.retrieved_chunks],
            "citations": [
                {
                    "source_index": c.source_index,
                    "chunk_id": c.chunk_id,
                    "quote": c.quote,
                    "source_type": c.source_type,
                }
                for c in self.citations
            ],
            "reliability": {
                "citation_score": self.reliability.citation_score if self.reliability else None,
                "grounding_score": self.reliability.grounding_score if self.reliability else None,
                "confidence": self.reliability.confidence if self.reliability else None,
                "unsupported_ratio": self.reliability.unsupported_ratio
                if self.reliability
                else None,
                "should_abstain": self.reliability.should_abstain if self.reliability else None,
                "abstention_reason": self.reliability.abstention_reason
                if self.reliability
                else None,
                "verdict": self.reliability.verdict if self.reliability else None,
                "sources_used": self.reliability.sources_used if self.reliability else [],
                "details": self.reliability.details if self.reliability else None,
            },
            "latency_ms": self.latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "model": self.model,
            "metadata": self.metadata,
        }
