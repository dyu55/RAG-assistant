"""RAG orchestration: process, retrieve, correct, generate, verify, record."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter
from uuid import uuid4

from core.crag import CRAGEvaluator
from core.generator import Generator
from core.models import GeneratedAnswer
from core.query_handler import QueryHandler
from core.reliability import ReliabilityChecker, ReliabilityReport
from core.result import ABSTENTION_MESSAGE as ABSTENTION_MESSAGE
from core.result import PipelineResult as PipelineResult
from core.retrieval import HybridRetrieval
from core.retriever import Retriever, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class Pipeline:
    """Coordinate replaceable stages, keeping request state local to each run."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        reliability_checker: ReliabilityChecker | None = None,
        query_handler: QueryHandler | None = None,
        query_logger=None,
        graph_retriever=None,
        router=None,
        crag_evaluator=None,
        semantic_cache=None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.reliability_checker = reliability_checker or ReliabilityChecker()
        self.query_handler = query_handler
        self.query_logger = query_logger
        self.graph_retriever = graph_retriever
        self.router = router
        self.semantic_cache = semantic_cache
        self.crag_evaluator = crag_evaluator or CRAGEvaluator()
        self._cache_namespace = uuid4().hex

    def invalidate_cache(self) -> None:
        """Invalidate this pipeline's answers after documents or configuration change."""
        self._cache_namespace = uuid4().hex

    @staticmethod
    @contextmanager
    def _timed(result: PipelineResult, stage: str):
        start = perf_counter()
        try:
            yield
        finally:
            result.latency_ms[stage] = round((perf_counter() - start) * 1000, 1)

    def run(
        self,
        query: str,
        top_k: int | None = None,
        temperature: float = 0.3,
        enable_rewrite: bool = True,
        enable_reranking: bool = False,
        one_shot_global: bool = True,
        enable_cache: bool = True,
    ) -> PipelineResult:
        """Answer a question and retain evidence and diagnostics on every return path."""
        start = perf_counter()
        result = PipelineResult(query=query, model=self.generator.provider.get_model_name())
        namespace = repr(
            (
                self._cache_namespace,
                result.model,
                top_k,
                temperature,
                enable_rewrite,
                enable_reranking,
                one_shot_global,
            )
        )
        if enable_cache:
            cached = self._load_cached(query, namespace)
            if cached is not None:
                return self._finish(cached, start)

        retrieval = HybridRetrieval(self.retriever, self.graph_retriever, self.router)
        retrieval_query = self._process_query(query, enable_rewrite, result)
        with self._timed(result, "route"):
            route = retrieval.decide_route(retrieval_query)
            result.route_mode = route.mode.value
            result.route_confidence = route.confidence
            result.route_reason = route.reason

        with self._timed(result, "retrieval"):
            try:
                chunks, timings = retrieval.retrieve(
                    retrieval_query,
                    route,
                    top_k,
                    enable_reranking,
                    one_shot_global,
                )
                result.retrieved_chunks = chunks
                result.latency_ms.update(timings)
            except Exception as exc:
                logger.error("Retrieval failed: %s", exc)

        self._correct_retrieval(retrieval_query, retrieval, result)
        self._generate(query, temperature, result)
        self._verify(result)
        self._finish(result, start)
        if enable_cache:
            self._store_cached(query, namespace, result)
        return result

    def _process_query(self, query, enable_rewrite, result):
        if self.query_handler is None or not enable_rewrite:
            return query
        with self._timed(result, "query_processing"):
            try:
                result.processed_query = self.query_handler.process(query, enable_rewrite=True)
                return result.processed_query.effective_query
            except Exception as exc:
                logger.warning("Query processing failed, using original: %s", exc)
                return query

    def _correct_retrieval(self, query, retrieval, result):
        with self._timed(result, "crag_eval"):
            try:
                evaluation = self.crag_evaluator.evaluate(query, result.retrieved_chunks)
                result.metadata.update(
                    crag_action=evaluation.action.value,
                    crag_confidence=evaluation.confidence,
                )
                if evaluation.is_ambiguous and evaluation.suggested_queries:
                    corrective = []
                    for sub_query in evaluation.suggested_queries:
                        corrective.extend(retrieval.vector(sub_query, top_k=2))
                    if corrective:
                        result.retrieved_chunks = reciprocal_rank_fusion(
                            [result.retrieved_chunks, corrective]
                        )
            except Exception as exc:
                logger.warning("CRAG evaluation failed: %s", exc)

    def _generate(self, query, temperature, result):
        with self._timed(result, "generation"):
            try:
                result.generated = self.generator.generate(
                    query=query,
                    chunks=result.retrieved_chunks,
                    temperature=temperature,
                )
            except Exception as exc:
                logger.error("Generation failed: %s", exc)
                result.metadata["generation_error"] = str(exc)
                result.generated = GeneratedAnswer(
                    answer="An error occurred during generation.",
                    self_confidence=0.0,
                )
            result.answer = result.generated.answer
            result.citations = result.generated.citations

    def _verify(self, result):
        with self._timed(result, "reliability"):
            try:
                result.reliability = self.reliability_checker.check(
                    answer=result.generated,
                    chunks=result.retrieved_chunks,
                )
            except Exception as exc:
                logger.error("Reliability check failed: %s", exc)
                result.reliability = ReliabilityReport(
                    confidence=0.0,
                    should_abstain=True,
                    abstention_reason=f"Reliability check error: {exc}",
                )
            if "generation_error" in result.metadata:
                result.reliability.should_abstain = True
                result.reliability.abstention_reason = "Answer generation failed."
            result.should_abstain = result.reliability.should_abstain
            if result.should_abstain:
                result.abstention_message = (
                    f"{ABSTENTION_MESSAGE}\n\n**Reason:** {result.reliability.abstention_reason}"
                )

    def _finish(self, result, start):
        result.total_latency_ms = round((perf_counter() - start) * 1000, 1)
        if self.query_logger is not None:
            try:
                self.query_logger.log(result)
            except Exception as exc:
                logger.error("Logging failed: %s", exc)
        return result

    def _load_cached(self, query, namespace):
        if self.semantic_cache is None:
            return None
        try:
            payload, similarity, hit_type = self.semantic_cache.get(query, namespace=namespace)
            if hit_type == "miss" or not payload:
                return None
            snapshot = payload.get("result")
            # Old or incomplete entries cannot bypass the evidence checks.
            if not isinstance(snapshot, PipelineResult) or snapshot.reliability is None:
                return None
            if snapshot.should_abstain or snapshot.reliability.should_abstain:
                return None
            result = deepcopy(snapshot)
            result.query = query
            result.latency_ms = {}
            result.metadata.update(cache_hit=hit_type, cache_similarity=similarity)
            return result
        except Exception as exc:
            logger.warning("Cache lookup failed, running pipeline: %s", exc)
            return None

    def _store_cached(self, query, namespace, result):
        if self.semantic_cache is None or result.should_abstain or not result.answer:
            return
        try:
            self.semantic_cache.put(
                query=query,
                query_embedding=[],
                namespace=namespace,
                payload={"result": deepcopy(result), "total_latency_ms": result.total_latency_ms},
            )
        except Exception as exc:
            logger.warning("Cache population failed: %s", exc)
