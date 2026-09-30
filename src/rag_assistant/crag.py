"""
Corrective RAG (CRAG) Retrieval Evaluation & Dynamic Knowledge Refinement.
Implements 2026 SOTA CRAG architecture (Yan et al.):
1. Retrieval Evaluator: Evaluates query-document relevance into CORRECT, AMBIGUOUS, or INCORRECT actions.
2. Corrective Query Rewriting: Generates broadened fallback queries when retrieval confidence is low.
3. Knowledge Strip Refinement: Decomposes evidence into granular knowledge strips, filtering irrelevant strips
   while strictly preserving source text for citation verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .models import Evidence
from .text import normalize, tokens

CRAGAction = Literal["correct", "ambiguous", "incorrect"]


@dataclass
class CRAGEvaluation:
    action: CRAGAction
    confidence: float
    reasoning: str
    fallback_query: str = ""


def evaluate_retrieval(
    query: str,
    evidence: list[Evidence],
    upper_threshold: float = 0.60,
    lower_threshold: float = 0.25,
) -> CRAGEvaluation:
    """Evaluate retrieval quality and determine the corrective action."""
    if not evidence or not query.strip():
        return CRAGEvaluation(
            action="incorrect",
            confidence=0.0,
            reasoning="No evidence retrieved or query is empty.",
            fallback_query=generate_corrective_query(query, evidence),
        )

    q_tokens = set(tokens(query))
    if not q_tokens:
        return CRAGEvaluation(
            action="incorrect",
            confidence=0.0,
            reasoning="Query has no meaningful lexical tokens.",
            fallback_query="",
        )

    # 1. Top relevance score
    max_relevance = max(e.relevance for e in evidence)

    # 2. Average relevance of top-min(3, len(evidence))
    top_n = min(3, len(evidence))
    avg_top_relevance = sum(e.relevance for e in evidence[:top_n]) / top_n

    # 3. Query token coverage across all retrieved evidence
    all_evidence_tokens = set(tokens(" ".join(e.text for e in evidence)))
    coverage = len(q_tokens & all_evidence_tokens) / len(q_tokens)

    # Composite confidence score
    confidence = round(0.50 * max_relevance + 0.25 * avg_top_relevance + 0.25 * coverage, 4)
    confidence = min(1.0, max(0.0, confidence))

    if confidence >= upper_threshold:
        return CRAGEvaluation(
            action="correct",
            confidence=confidence,
            reasoning=f"High retrieval confidence ({confidence:.2f}); knowledge verified.",
        )
    elif confidence >= lower_threshold:
        return CRAGEvaluation(
            action="ambiguous",
            confidence=confidence,
            reasoning=f"Ambiguous retrieval confidence ({confidence:.2f}); refining knowledge and broadening context.",
            fallback_query=generate_corrective_query(query, evidence),
        )
    else:
        return CRAGEvaluation(
            action="incorrect",
            confidence=confidence,
            reasoning=f"Low retrieval confidence ({confidence:.2f}); passages irrelevant or insufficient.",
            fallback_query=generate_corrective_query(query, evidence),
        )


def generate_corrective_query(query: str, evidence: list[Evidence]) -> str:
    """Reformulate query by removing conversational noise, stop words, and punctuation."""
    clean_q = normalize(query)
    pattern = re.compile(
        r"^(please|could you please|can you please|could you|can you|tell me about|explain to me|explain|i want to know|what is|how does|how do|why does)\s+",
        re.IGNORECASE,
    )
    while True:
        subbed = pattern.sub("", clean_q).strip()
        if subbed == clean_q:
            break
        clean_q = subbed

    stopwords = {
        "to",
        "me",
        "how",
        "does",
        "do",
        "you",
        "could",
        "can",
        "please",
        "explain",
        "about",
    }
    words = [t for t in tokens(clean_q) if len(t) > 2 and t not in stopwords]
    return " ".join(words) if words else clean_q


def decompose_into_knowledge_strips(text: str) -> list[str]:
    """Decompose passage into granular knowledge strips (sentences or bullet points)."""
    raw_strips = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [s.strip() for s in raw_strips if len(s.strip()) >= 15]


def refine_knowledge_strips(
    evidence: list[Evidence],
    query: str,
    min_strip_overlap: float = 0.05,
) -> list[Evidence]:
    """Refine retrieved evidence by decomposing into knowledge strips and filtering noise."""
    if not evidence or not query.strip():
        return evidence

    q_tokens = set(tokens(query))
    if not q_tokens:
        return evidence

    refined_list: list[Evidence] = []
    for ev in evidence:
        text_to_refine = ev.compressed_text or ev.text
        strips = decompose_into_knowledge_strips(text_to_refine)
        if not strips:
            refined_list.append(ev)
            continue

        salient_strips = []
        for strip in strips:
            s_tokens = set(tokens(strip))
            overlap = len(q_tokens & s_tokens) / len(q_tokens)
            if overlap >= min_strip_overlap:
                salient_strips.append(strip)

        if salient_strips:
            refined_text = " ".join(salient_strips)
            ratio = round(len(refined_text) / max(1, len(ev.text)), 3)
            refined_list.append(
                ev.model_copy(
                    update={
                        "compressed_text": refined_text,
                        "compression_ratio": ratio,
                    }
                )
            )
        else:
            refined_list.append(ev)

    return refined_list
