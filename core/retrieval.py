"""Routing and concurrent retrieval, independent of generation and caching."""

from __future__ import annotations

import inspect
import logging
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from core.models import RetrievedChunk
from core.retriever import reciprocal_rank_fusion
from graph.router import RouteDecision, RouteMode

logger = logging.getLogger(__name__)


class HybridRetrieval:
    """Combine available retrieval paths while isolating backend failures."""

    def __init__(self, retriever, graph_retriever=None, router=None):
        self.retriever = retriever
        self.graph_retriever = graph_retriever
        self.router = router

    def decide_route(self, query: str) -> RouteDecision:
        if self.router is None:
            return RouteDecision(
                mode=RouteMode.LOCAL,
                reason="Default local route (No router configured)",
            )
        try:
            route = self.router.route(query)
        except Exception as exc:
            logger.warning("Routing failed, falling back to vector retrieval: %s", exc)
            return RouteDecision(RouteMode.OFF, confidence=0.0, reason=f"Router error: {exc}")
        if self.graph_retriever is None and route.mode == RouteMode.GLOBAL:
            return RouteDecision(RouteMode.OFF, reason="Graph unavailable; using vector retrieval")
        return route

    def vector(self, query: str, top_k=None, enable_reranking=False) -> list[RetrievedChunk]:
        try:
            return (
                self.retriever.retrieve(query, top_k=top_k, enable_reranking=enable_reranking) or []
            )
        except Exception as exc:
            logger.warning("Vector retrieval failed: %s", exc)
            return []

    def _global(self, query: str, one_shot: bool) -> list[RetrievedChunk]:
        search = self.graph_retriever.global_search
        if "one_shot" in inspect.signature(search).parameters:
            return search(query, one_shot=one_shot)
        return search(query)

    @staticmethod
    def _timed_call(name, callback):
        start = perf_counter()
        try:
            return callback() or [], round((perf_counter() - start) * 1000, 1)
        except Exception as exc:
            logger.warning("%s retrieval failed: %s", name, exc)
            return [], round((perf_counter() - start) * 1000, 1)

    def retrieve(
        self,
        query: str,
        route: RouteDecision,
        top_k=None,
        enable_reranking=False,
        one_shot_global=True,
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Measure backend work inside each worker, then fuse rankings.

        Network deadlines belong to the backend clients. Waiting on a future
        with a timeout cannot cancel an in-flight network request.
        """
        callbacks = {}
        if route.run_vector:
            callbacks["vector"] = lambda: self.vector(query, top_k, enable_reranking)
        if self.graph_retriever is not None:
            if route.run_graph_local:
                callbacks["graph_local"] = lambda: self.graph_retriever.local_search(query)
            if route.run_graph_global:
                callbacks["graph_global"] = lambda: self._global(query, one_shot_global)

        if not callbacks:
            return [], {}
        results, timings = {}, {}
        with ThreadPoolExecutor(max_workers=len(callbacks)) as pool:
            futures = {
                name: pool.submit(self._timed_call, name, callback)
                for name, callback in callbacks.items()
            }
            for name, future in futures.items():
                results[name], timings[f"sub_{name}"] = future.result()

        details = [results[name] for name in ("vector", "graph_local") if results.get(name)]
        fused = reciprocal_rank_fusion(details)
        merged = {}
        for chunk in [*results.get("graph_global", []), *fused]:
            merged.setdefault(chunk.chunk_id, chunk)
        return list(merged.values()), timings
