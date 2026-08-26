"""
Pipeline Orchestrator.
Wires all layers together: retrieval → generation → reliability → logging.
Single entry point: pipeline.run(query) → PipelineResult.

Hybrid (GraphRAG) mode
----------------------
When a `GraphRetriever` is provided, the pipeline runs vector and graph
retrieval in parallel (they are both I/O bound). The graph retrieval may
be skipped when the `QueryRouter` decides the question is purely local.

The merged chunks preserve their `source` field so the generator and the
reliability layer can distinguish vector hits (`[V N]`) from graph hits
(`[G N]`) and community reports (`[C N]`).
"""
from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor
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

    # Routing & observability
    route_mode: str = "vector-only"
    route_confidence: float = 0.0
    route_reason: str = ""
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
                "unsupported_ratio": self.reliability.unsupported_ratio if self.reliability else None,
                "should_abstain": self.reliability.should_abstain if self.reliability else None,
                "abstention_reason": self.reliability.abstention_reason if self.reliability else None,
                "verdict": self.reliability.verdict if self.reliability else None,
                "sources_used": self.reliability.sources_used if self.reliability else [],
                "details": self.reliability.details if self.reliability else None,
            },
            "latency_ms": self.latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "model": self.model,
        }


class Pipeline:
    """
    Main RAG pipeline orchestrator.
    query → query_rewrite → routing → (vector ‖ graph) → reranking → generation → reliability → result
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        reliability_checker: ReliabilityChecker | None = None,
        query_handler: QueryHandler | None = None,
        query_logger=None,
        # Optional GraphRAG components. When `graph_retriever` is None
        # the pipeline behaves exactly as it did before this change.
        graph_retriever=None,
        router=None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.reliability_checker = reliability_checker or ReliabilityChecker()
        self.query_handler = query_handler
        self.query_logger = query_logger
        self.graph_retriever = graph_retriever
        self.router = router

    def run(
        self,
        query: str,
        top_k: int | None = None,
        temperature: float = 0.3,
        enable_rewrite: bool = True,
        enable_reranking: bool = False,
        one_shot_global: bool = True,
    ) -> PipelineResult:
        """
        Execute the full RAG pipeline.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.
            temperature: LLM temperature.
            enable_rewrite: Whether to rewrite vague queries.
            enable_reranking: Whether to LLM-rerank retrieved chunks.
            one_shot_global: Whether to use fast one-shot synthesis for global community search.

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

        # ── Layer 0.5: Route decision ─────────────────────────────────────
        t0 = time.time()
        route = self._decide_route(retrieval_query)
        result.route_mode = route.mode.value
        result.route_confidence = route.confidence
        result.route_reason = route.reason
        result.latency_ms["route"] = round((time.time() - t0) * 1000, 1)

        # ── Layer 1: Retrieval (hybrid) ───────────────────────────────────
        t0 = time.time()
        chunks: list[RetrievedChunk] = []
        try:
            chunks = self._hybrid_retrieve(
                retrieval_query,
                top_k=top_k,
                enable_reranking=enable_reranking,
                route=route,
                one_shot_global=one_shot_global,
                result=result,
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
            f"route={result.route_mode} latency={result.total_latency_ms}ms (route={result.latency_ms.get('route', 0)}ms, retrieval={result.latency_ms.get('retrieval', 0)}ms)"
        )

        return result

    # ── Hybrid retrieval helpers ─────────────────────────────────────────────

    def _decide_route(self, query: str):
        """Invoke the QueryRouter, or fall back to LOCAL if unconfigured."""
        if self.router is not None:
            return self.router.route(query)
        try:
            from graph.router import RouteDecision, RouteMode
            return RouteDecision(
                mode=RouteMode.LOCAL,
                confidence=1.0,
                reason="Default local route (No router configured)",
            )
        except Exception as e:
            from graph.router import RouteDecision, RouteMode
            return RouteDecision(
                mode=RouteMode.LOCAL,
                confidence=0.0,
                reason=f"Router error: {e}",
            )

    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int | None,
        enable_reranking: bool,
        route,
        one_shot_global: bool = True,
        result: PipelineResult | None = None,
    ) -> list[RetrievedChunk]:
        """Run vector + graph retrievers in parallel when applicable."""
        tasks: dict = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            if route.run_vector:
                tasks["vector"] = pool.submit(
                    self._safe_vector_retrieve, query, top_k, enable_reranking
                )
            if route.run_graph_local and self.graph_retriever is not None:
                tasks["graph_local"] = pool.submit(
                    self._safe_graph_local, query
                )
            if route.run_graph_global and self.graph_retriever is not None:
                tasks["graph_global"] = pool.submit(
                    self._safe_graph_global, query, one_shot_global
                )

            results: dict[str, list[RetrievedChunk]] = {}
            for name, fut in tasks.items():
                t_sub = time.time()
                try:
                    results[name] = fut.result(timeout=60) or []
                except Exception as e:
                    logger.warning(f"{name} retrieval failed: {e}")
                    results[name] = []
                if result is not None:
                    result.latency_ms[f"sub_{name}"] = round((time.time() - t_sub) * 1000, 1)

        # Community chunks (Global Search) are macro-synthesized answers, prioritized first
        global_chunks = results.get("graph_global", [])
        vector_chunks = results.get("vector", [])
        local_chunks = results.get("graph_local", [])

        # Fuse vector and local graph rankings using Reciprocal Rank Fusion (RRF)
        from core.retriever import reciprocal_rank_fusion
        detail_lists = [l for l in [vector_chunks, local_chunks] if l]
        fused_details = reciprocal_rank_fusion(detail_lists) if detail_lists else []

        merged: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for c in global_chunks:
            if c.chunk_id not in seen_ids:
                merged.append(c)
                seen_ids.add(c.chunk_id)

        for c in fused_details:
            if c.chunk_id not in seen_ids:
                merged.append(c)
                seen_ids.add(c.chunk_id)

        return merged

    def _safe_vector_retrieve(self, query, top_k, enable_reranking) -> list[RetrievedChunk]:
        try:
            return self.retriever.retrieve(
                query, top_k=top_k, enable_reranking=enable_reranking
            )
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return []

    def _safe_graph_local(self, query) -> list[RetrievedChunk]:
        try:
            return self.graph_retriever.local_search(query)
        except Exception as e:
            logger.warning(f"Graph local retrieval failed: {e}")
            return []

    def _safe_graph_global(self, query, one_shot: bool = True) -> list[RetrievedChunk]:
        try:
            import inspect
            sig = inspect.signature(self.graph_retriever.global_search)
            if "one_shot" in sig.parameters:
                return self.graph_retriever.global_search(query, one_shot=one_shot)
            return self.graph_retriever.global_search(query)
        except Exception as e:
            logger.warning(f"Graph global retrieval failed: {e}")
            return []