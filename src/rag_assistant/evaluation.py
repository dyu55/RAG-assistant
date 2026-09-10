"""
RAG Triad & Groundedness Evaluation Engine.
Evaluates end-to-end RAG response quality across the standard three pillars:
1. Context Relevance: Did retrieval pull passages containing the necessary query information?
2. Groundedness / Faithfulness: Are generated claims strictly substantiated by the cited quotes?
3. Answer Relevance: Does the synthesized answer directly address the user's inquiry?
"""

from __future__ import annotations

from .models import Claim, Evidence, TriadMetrics
from .text import normalize, tokens


def compute_context_relevance(query: str, evidence: list[Evidence]) -> float:
    """
    Measure how relevant the retrieved context is to the question.
    Combines lexical term coverage with the retrieval engine's relevance scores.
    """
    if not evidence or not query.strip():
        return 0.0

    q_tokens = set(tokens(query))
    if not q_tokens:
        return 0.0

    combined_text = " ".join(e.text for e in evidence)
    c_tokens = set(tokens(combined_text))

    coverage = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    avg_relevance = sum(e.relevance for e in evidence) / len(evidence)

    # 60% lexical coverage + 40% retrieval score
    score = 0.6 * coverage + 0.4 * min(1.0, avg_relevance)
    return round(max(0.0, min(1.0, score)), 3)


def compute_groundedness(claims: list[Claim], evidence: list[Evidence]) -> float:
    """
    Measure the degree to which claims are strictly anchored in retrieved passages.
    Penalizes missing quotes, quotes not found in source text, and low lexical overlap.
    """
    if not claims:
        return 0.0

    sources = {item.source_id: item for item in evidence}
    scores = []

    for claim in claims:
        source = sources.get(claim.source_id)
        if not source:
            scores.append(0.0)
            continue

        # Exact normalized substring check
        if normalize(claim.quote) not in normalize(source.text):
            scores.append(0.0)
            continue

        # Lexical support check: claim words grounded in the quote
        c_words = set(tokens(claim.text))
        q_words = set(tokens(claim.quote))
        if not c_words:
            scores.append(0.0)
            continue

        overlap = len(c_words & q_words) / len(c_words)
        # Scale: overlap >= 0.45 is baseline acceptable, 0.8+ is fully grounded
        claim_score = min(1.0, overlap / 0.8)
        scores.append(claim_score)

    return round(sum(scores) / len(scores), 3) if scores else 0.0


def compute_answer_relevance(query: str, answer_text: str) -> float:
    """
    Measure how well the final synthesized answer addresses the user's question.
    """
    q_tokens = set(tokens(query))
    a_tokens = set(tokens(answer_text))

    if not q_tokens or not a_tokens:
        return 0.0

    overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
    return round(max(0.0, min(1.0, overlap)), 3)


def evaluate_triad(
    query: str,
    evidence: list[Evidence],
    answer_text: str,
    claims: list[Claim],
    status: str,
) -> TriadMetrics:
    """
    Execute full RAG Triad assessment and return structured metrics.
    """
    if status == "abstained":
        return TriadMetrics(
            context_relevance=compute_context_relevance(query, evidence),
            groundedness=1.0,  # Accurately abstained without hallucinating
            answer_relevance=0.0,
            composite_score=0.0,
            verdict="abstained",
        )

    c_rel = compute_context_relevance(query, evidence)
    grounded = compute_groundedness(claims, evidence)
    a_rel = compute_answer_relevance(query, answer_text)

    # Harmonic / weighted mean
    # Groundedness is critical (50%), Context Relevance (30%), Answer Relevance (20%)
    composite = 0.5 * grounded + 0.3 * c_rel + 0.2 * a_rel
    composite = round(max(0.0, min(1.0, composite)), 3)

    if composite >= 0.80:
        verdict = "EXCELLENT"
    elif composite >= 0.60:
        verdict = "GOOD"
    else:
        verdict = "NEEDS_REVIEW"

    return TriadMetrics(
        context_relevance=c_rel,
        groundedness=grounded,
        answer_relevance=a_rel,
        composite_score=composite,
        verdict=verdict,
    )
