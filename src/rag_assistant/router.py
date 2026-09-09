"""
Adaptive Retrieval Router (Dynamic Retrieval Routing).
Analyzes query characteristics (exact identifiers, relational queries, abstract semantics)
to dynamically select the optimal retrieval mode (keyword, vector, graph, or hybrid).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RetrievalMode = Literal["hybrid", "keyword", "vector", "graph"]


@dataclass(frozen=True)
class RouteDecision:
    """The outcome of an adaptive routing assessment."""

    mode: RetrievalMode
    confidence: float
    reasoning: str


class AdaptiveRouter:
    """
    Classifies user queries into optimal retrieval channels based on
    lexical precision, relational intent, and conceptual abstraction.
    """

    RELATIONAL_PATTERNS = (
        r"\b(connected to|relationship between|relates to|associated with|co-occurs? with)\b",
        r"\b(who worked with|network of|entities related to|linked with)\b",
    )
    EXACT_IDENTIFIER_PATTERNS = (
        r"[\"'].+?[\"']",  # Exact quotes
        r"\b[A-Z0-9_]{4,}\b",  # Constants / Error codes / API names like HTTP_404, JWT_SECRET
        r"\b\d+(\.\d+)?\b",  # Numeric constraints, versions, or IDs
        r"(/|\w+\.\w{2,4})",  # File paths or file extensions (e.g. main.py, /api/v1)
    )
    CONCEPTUAL_PATTERNS = (
        r"\b(concept|essence|philosophy|overview|broadly|in general|metaphor|intuition)\b",
        r"\b(what does it mean|explain the idea of|high-level)\b",
    )

    def route(self, query: str) -> RouteDecision:
        """Evaluate query characteristics and return an optimal retrieval mode."""
        clean_q = query.strip()
        if not clean_q:
            return RouteDecision(
                mode="hybrid",
                confidence=1.0,
                reasoning="Empty query defaults to hybrid retrieval.",
            )

        lower_q = clean_q.lower()

        # 1. Relational intent -> Knowledge Graph
        for pat in self.RELATIONAL_PATTERNS:
            if re.search(pat, lower_q):
                return RouteDecision(
                    mode="graph",
                    confidence=0.88,
                    reasoning="Query seeks entity relationships or network connections.",
                )

        # 2. Exact identifiers / code / numbers / quoted strings -> BM25 Keyword
        has_exact = any(re.search(pat, clean_q) for pat in self.EXACT_IDENTIFIER_PATTERNS)
        if has_exact and len(clean_q.split()) <= 8:
            return RouteDecision(
                mode="keyword",
                confidence=0.85,
                reasoning="Query targets specific identifiers, exact quotes, or precise codes.",
            )

        # 3. Highly abstract or conceptual queries -> Dense Vector
        for pat in self.CONCEPTUAL_PATTERNS:
            if re.search(pat, lower_q) and not has_exact:
                return RouteDecision(
                    mode="vector",
                    confidence=0.82,
                    reasoning="Query is abstract/conceptual, favoring dense semantic embedding matching.",
                )

        # 4. Default -> Hybrid (BM25 + Vector + Graph RRF)
        return RouteDecision(
            mode="hybrid",
            confidence=0.95,
            reasoning="Query benefits from multi-channel consensus (lexical + semantic + graph).",
        )


default_router = AdaptiveRouter()


def route_query(query: str) -> RouteDecision:
    """Convenience functional router interface."""
    return default_router.route(query)
