"""Regression coverage for evidence, fallback and cache boundaries."""

import json
from unittest.mock import Mock

import pytest

from core.cache import SemanticCache
from core.generator import Citation, GeneratedAnswer
from core.pipeline import Pipeline, PipelineResult
from core.reliability import ReliabilityChecker
from core.retriever import RetrievedChunk, reciprocal_rank_fusion
from graph.router import RouteDecision, RouteMode


@pytest.fixture
def pipeline():
    text = "Python is a programming language."
    retriever = Mock()
    retriever.retrieve.return_value = [RetrievedChunk("c1", text, 0.9)]
    generator = Mock()
    generator.provider.get_model_name.return_value = "test-model"
    generator.generate.return_value = GeneratedAnswer(
        answer="Python is a programming language [V 1].",
        citations=[Citation(1, "c1", text)],
        self_confidence=0.95,
    )
    return Pipeline(retriever, generator, query_logger=Mock(), semantic_cache=SemanticCache())


def test_fused_relevance_does_not_trigger_false_abstention(pipeline):
    result = pipeline.run("What is Python?")
    assert result.retrieved_chunks[0].score == 0.9
    assert result.retrieved_chunks[0].metadata["rrf_score"] < 0.1
    assert result.metadata["crag_action"] == "correct"
    assert result.reliability.confidence > 0.8
    assert not result.should_abstain


def test_cache_preserves_evidence_logs_hits_and_isolates_mutations(pipeline):
    first = pipeline.run("What is Python?")
    original_confidence = first.reliability.confidence
    first.citations.clear()
    first.reliability.confidence = 0.0
    second = pipeline.run("What is Python?")
    assert second.metadata["cache_hit"] == "exact"
    assert second.reliability.confidence == original_confidence
    assert second.citations[0].chunk_id == "c1"
    assert second.retrieved_chunks[0].text.startswith("Python")
    assert second.generated is not None
    assert second.route_mode == first.route_mode
    assert pipeline.generator.generate.call_count == 1
    assert pipeline.query_logger.log.call_count == 2
    assert json.loads(json.dumps(second.to_dict()))["metadata"]["cache_hit"] == "exact"


@pytest.mark.parametrize(
    "option",
    [
        {"top_k": 1},
        {"temperature": 0.9},
        {"enable_rewrite": False},
        {"enable_reranking": True},
        {"one_shot_global": False},
        {"enable_cache": False},
    ],
)
def test_cache_respects_run_options(pipeline, option):
    pipeline.run("What is Python?")
    result = pipeline.run("What is Python?", **option)
    assert "cache_hit" not in result.metadata
    assert pipeline.generator.generate.call_count == 2


def test_cache_invalidation_after_document_changes(pipeline):
    pipeline.run("What is Python?")
    pipeline.invalidate_cache()
    pipeline.run("What is Python?")
    assert pipeline.generator.generate.call_count == 2


def test_broken_cache_does_not_break_answering(pipeline):
    pipeline.semantic_cache = Mock()
    pipeline.semantic_cache.get.side_effect = RuntimeError("cache offline")
    pipeline.semantic_cache.put.side_effect = RuntimeError("cache offline")
    assert not pipeline.run("What is Python?").should_abstain


def test_shared_cache_does_not_cross_pipeline_boundaries(pipeline):
    pipeline.run("What is Python?")
    another = Pipeline(
        pipeline.retriever, pipeline.generator, semantic_cache=pipeline.semantic_cache
    )
    assert "cache_hit" not in another.run("What is Python?").metadata


def test_router_failure_falls_back_to_vector(pipeline):
    pipeline.router = Mock()
    pipeline.router.route.side_effect = RuntimeError("router offline")
    result = pipeline.run("What is Python?")
    assert result.route_mode == "off"
    assert not result.should_abstain


def test_global_route_without_graph_falls_back_to_vector(pipeline):
    pipeline.router = Mock()
    pipeline.router.route.return_value = RouteDecision(RouteMode.GLOBAL)
    result = pipeline.run("What is Python?")
    assert result.retrieved_chunks
    assert not result.should_abstain


def test_generation_failure_is_never_cached(pipeline):
    pipeline.generator.generate.side_effect = RuntimeError("provider offline")
    assert pipeline.run("What is Python?").should_abstain
    assert pipeline.semantic_cache.size() == 0


def test_reliability_failure_abstains_and_is_not_cached(pipeline):
    pipeline.reliability_checker = Mock(spec=ReliabilityChecker)
    pipeline.reliability_checker.check.side_effect = RuntimeError("checker failed")
    assert pipeline.run("What is Python?").should_abstain
    assert pipeline.semantic_cache.size() == 0


def test_duplicate_chunk_in_one_ranking_does_not_inflate_consensus():
    first = RetrievedChunk("a", "A", 0.8)
    second = RetrievedChunk("b", "B", 0.7)
    fused = reciprocal_rank_fusion([[first, second, second, second]])
    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]


def test_compatibility_result_import():
    from core.result import PipelineResult as Result

    assert PipelineResult is Result
