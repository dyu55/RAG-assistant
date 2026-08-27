"""Corrective Retrieval-Augmented Generation (CRAG) Evaluator.

Implements the 2026 SOTA CRAG pattern:
1. Retrieval Evaluator: Evaluates candidate chunks before generation to prevent hallucinations.
2. Three-way Classification:
   - CORRECT (confidence >= 0.7): Evidence is robust, proceed directly to generation.
   - AMBIGUOUS (0.35 <= confidence < 0.7): Evidence is partial or noisy; trigger query expansion.
   - INCORRECT (confidence < 0.35): Retrieved context is insufficient; trigger fallback/abstention.
3. Query Decomposition & Correction: Generates targeted sub-queries to rescue ambiguous retrievals.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class CRAGAction(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class CRAGEvaluation:
    """Assessment result of retrieved context quality before generation."""
    action: CRAGAction
    confidence: float
    reason: str
    relevance_scores: list[float] = field(default_factory=list)
    suggested_queries: list[str] = field(default_factory=list)

    @property
    def is_correct(self) -> bool:
        return self.action == CRAGAction.CORRECT

    @property
    def is_ambiguous(self) -> bool:
        return self.action == CRAGAction.AMBIGUOUS

    @property
    def is_incorrect(self) -> bool:
        return self.action == CRAGAction.INCORRECT


class CRAGEvaluator:
    """Evaluates retrieval quality and determines whether corrective action is needed."""

    def __init__(
        self,
        correct_threshold: float = 0.65,
        incorrect_threshold: float = 0.35,
    ):
        self.correct_threshold = correct_threshold
        self.incorrect_threshold = incorrect_threshold

    def evaluate(self, query: str, chunks: Sequence[RetrievedChunk]) -> CRAGEvaluation:
        """Classify retrieval candidate pool into CORRECT, AMBIGUOUS, or INCORRECT."""
        if not chunks or not query.strip():
            return CRAGEvaluation(
                action=CRAGAction.INCORRECT,
                confidence=0.0,
                reason="No chunks retrieved for evaluation.",
                relevance_scores=[],
                suggested_queries=self.decompose_and_expand_query(query),
            )

        # Calculate semantic & keyword relevance scores
        scores = []
        q_tokens = set(re.findall(r"\w+", query.lower()))
        q_tokens = {t for t in q_tokens if len(t) > 2}

        for c in chunks:
            # Base score from retriever
            base_score = float(c.effective_score)
            
            # Keyword coverage bonus
            chunk_lower = c.text.lower()
            overlap = sum(1 for t in q_tokens if t in chunk_lower) / max(len(q_tokens), 1) if q_tokens else 0.5
            
            # Blended relevance score: weighted blend bounded by base score
            blended = round(max(base_score * 0.85, 0.7 * base_score + 0.3 * overlap), 4)
            scores.append(blended)

        scores.sort(reverse=True)
        top_score = scores[0]
        avg_top_score = sum(scores[:2]) / min(len(scores), 2)

        # Classification rules
        if top_score >= self.correct_threshold or avg_top_score >= (self.correct_threshold - 0.05):
            return CRAGEvaluation(
                action=CRAGAction.CORRECT,
                confidence=round(top_score, 3),
                reason=f"High retrieval confidence ({top_score:.2f} >= {self.correct_threshold}).",
                relevance_scores=scores,
                suggested_queries=[],
            )
        elif top_score >= self.incorrect_threshold:
            suggested = self.decompose_and_expand_query(query)
            return CRAGEvaluation(
                action=CRAGAction.AMBIGUOUS,
                confidence=round(top_score, 3),
                reason=f"Partial/ambiguous relevance ({top_score:.2f}); corrective query expansion recommended.",
                relevance_scores=scores,
                suggested_queries=suggested,
            )
        else:
            suggested = self.decompose_and_expand_query(query)
            return CRAGEvaluation(
                action=CRAGAction.INCORRECT,
                confidence=round(top_score, 3),
                reason=f"Low retrieval quality ({top_score:.2f} < {self.incorrect_threshold}).",
                relevance_scores=scores,
                suggested_queries=suggested,
            )

    def decompose_and_expand_query(self, query: str) -> list[str]:
        """Decompose complex query into atomic search components for corrective retrieval."""
        q = query.strip()
        if not q:
            return []

        expanded = []
        
        # 1. Strip question words
        clean_q = re.sub(r"^(what is|how does|why does|where is|explain|summarize|tell me about)\s+", "", q, flags=re.IGNORECASE)
        if clean_q and clean_q.lower() != q.lower():
            expanded.append(clean_q.strip())

        # 2. Extract key noun phrases / capitalized terms
        capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", q)
        if len(capitalized) >= 2:
            expanded.append(" ".join(capitalized))

        # 3. Check for comparison connectors (vs, and, between)
        if " and " in q.lower():
            parts = re.split(r"\s+and\s+", q, flags=re.IGNORECASE)
            expanded.extend([p.strip() for p in parts if len(p.strip()) > 3])
        elif " vs " in q.lower():
            parts = re.split(r"\s+vs\s+", q, flags=re.IGNORECASE)
            expanded.extend([p.strip() for p in parts if len(p.strip()) > 3])

        # Deduplicate while preserving order
        unique_expanded = []
        for item in expanded:
            if item and item not in unique_expanded and item.lower() != q.lower():
                unique_expanded.append(item)

        return unique_expanded[:3]
