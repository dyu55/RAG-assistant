"""
Evaluation Metrics.
Computes quality metrics from pipeline results for analysis and benchmarking.

Metrics:
- Faithfulness: % of answer claims supported by context
- Answer Relevancy: semantic overlap between question and answer
- Citation Coverage: % of answer sentences that have citations
- Grounding Rate: % of citations verified against sources
- Abstention Rate: how often the system refuses to answer
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics computed from a single pipeline result."""
    faithfulness: float       # 1 - unsupported_ratio
    citation_coverage: float  # citation_score from reliability
    grounding_rate: float     # grounding_score from reliability
    confidence: float         # overall confidence
    abstained: bool           # whether system abstained
    latency_ms: float         # total pipeline latency

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "citation_coverage": round(self.citation_coverage, 3),
            "grounding_rate": round(self.grounding_rate, 3),
            "confidence": round(self.confidence, 3),
            "abstained": self.abstained,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class AggregateMetrics:
    """Aggregated metrics over multiple queries."""
    total_queries: int
    avg_faithfulness: float
    avg_citation_coverage: float
    avg_grounding_rate: float
    avg_confidence: float
    abstention_rate: float
    avg_latency_ms: float


def compute_query_metrics(result_dict: dict) -> QueryMetrics:
    """Compute metrics from a single logged pipeline result (dict form)."""
    rel = result_dict.get("reliability", {}) or {}

    unsupported_ratio = rel.get("unsupported_ratio", 0.0) or 0.0
    faithfulness = 1.0 - unsupported_ratio

    return QueryMetrics(
        faithfulness=faithfulness,
        citation_coverage=rel.get("citation_score", 0.0) or 0.0,
        grounding_rate=rel.get("grounding_score", 0.0) or 0.0,
        confidence=rel.get("confidence", 0.0) or 0.0,
        abstained=result_dict.get("should_abstain", False),
        latency_ms=result_dict.get("total_latency_ms", 0.0) or 0.0,
    )


def compute_aggregate_metrics(results: list[dict]) -> AggregateMetrics:
    """Compute aggregate metrics over a list of logged results."""
    if not results:
        return AggregateMetrics(
            total_queries=0,
            avg_faithfulness=0.0,
            avg_citation_coverage=0.0,
            avg_grounding_rate=0.0,
            avg_confidence=0.0,
            abstention_rate=0.0,
            avg_latency_ms=0.0,
        )

    metrics_list = [compute_query_metrics(r) for r in results]
    n = len(metrics_list)

    return AggregateMetrics(
        total_queries=n,
        avg_faithfulness=round(sum(m.faithfulness for m in metrics_list) / n, 3),
        avg_citation_coverage=round(sum(m.citation_coverage for m in metrics_list) / n, 3),
        avg_grounding_rate=round(sum(m.grounding_rate for m in metrics_list) / n, 3),
        avg_confidence=round(sum(m.confidence for m in metrics_list) / n, 3),
        abstention_rate=round(sum(1 for m in metrics_list if m.abstained) / n, 3),
        avg_latency_ms=round(sum(m.latency_ms for m in metrics_list) / n, 1),
    )
