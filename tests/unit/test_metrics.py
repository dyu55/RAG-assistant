"""
Unit tests for evaluation/metrics.py.
Tests calculation of faithfulness, citation coverage, grounding rate, confidence,
abstention rate, and aggregated benchmark statistics.
"""

from __future__ import annotations

from evaluation.metrics import (
    AggregateMetrics,
    QueryMetrics,
    compute_aggregate_metrics,
    compute_query_metrics,
)


class TestQueryMetrics:
    def test_compute_query_metrics_normal(self):
        logged_result = {
            "query": "What is RAG?",
            "answer": "RAG stands for Retrieval-Augmented Generation [V1].",
            "should_abstain": False,
            "total_latency_ms": 245.8,
            "reliability": {
                "unsupported_ratio": 0.1,
                "citation_score": 0.9,
                "grounding_score": 0.85,
                "confidence": 0.88,
            },
        }

        m = compute_query_metrics(logged_result)
        assert isinstance(m, QueryMetrics)
        assert round(m.faithfulness, 2) == 0.90
        assert m.citation_coverage == 0.9
        assert m.grounding_rate == 0.85
        assert m.confidence == 0.88
        assert m.abstained is False
        assert m.latency_ms == 245.8

        d = m.to_dict()
        assert d["faithfulness"] == 0.9
        assert d["citation_coverage"] == 0.9
        assert d["grounding_rate"] == 0.85
        assert d["confidence"] == 0.88
        assert d["abstained"] is False
        assert d["latency_ms"] == 245.8

    def test_compute_query_metrics_abstained_and_missing_keys(self):
        logged_result = {
            "query": "Unknown topic?",
            "should_abstain": True,
            "reliability": None,
        }

        m = compute_query_metrics(logged_result)
        assert m.faithfulness == 1.0
        assert m.citation_coverage == 0.0
        assert m.grounding_rate == 0.0
        assert m.confidence == 0.0
        assert m.abstained is True
        assert m.latency_ms == 0.0


class TestAggregateMetrics:
    def test_compute_aggregate_metrics_empty(self):
        agg = compute_aggregate_metrics([])
        assert isinstance(agg, AggregateMetrics)
        assert agg.total_queries == 0
        assert agg.avg_faithfulness == 0.0
        assert agg.avg_citation_coverage == 0.0
        assert agg.avg_grounding_rate == 0.0
        assert agg.avg_confidence == 0.0
        assert agg.abstention_rate == 0.0
        assert agg.avg_latency_ms == 0.0

    def test_compute_aggregate_metrics_multiple(self):
        results = [
            {
                "should_abstain": False,
                "total_latency_ms": 100.0,
                "reliability": {
                    "unsupported_ratio": 0.2,
                    "citation_score": 0.8,
                    "grounding_score": 0.8,
                    "confidence": 0.8,
                },
            },
            {
                "should_abstain": True,
                "total_latency_ms": 200.0,
                "reliability": {
                    "unsupported_ratio": 0.0,
                    "citation_score": 0.6,
                    "grounding_score": 0.6,
                    "confidence": 0.4,
                },
            },
        ]

        agg = compute_aggregate_metrics(results)
        assert agg.total_queries == 2
        assert agg.avg_faithfulness == 0.9
        assert agg.avg_citation_coverage == 0.7
        assert agg.avg_grounding_rate == 0.7
        assert agg.avg_confidence == 0.6
        assert agg.abstention_rate == 0.5
        assert agg.avg_latency_ms == 150.0
