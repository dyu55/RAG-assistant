from __future__ import annotations

import math
import re
from collections import Counter

from .text import STOP_WORDS, entities, tokens


def synthesize_hypothetical_document(query: str) -> str:
    """Generate a declarative, domain-aligned hypothetical document passage

    to bridge the semantic gap between questions and technical corpus texts.
    """
    clean_q = query.strip()
    lower_q = clean_q.lower()

    # Detect extracted entities to anchor domain synthesis
    extracted = entities(clean_q)
    entity_str = ", ".join(e.title() for e in extracted[:3]) if extracted else ""

    # Rule-based declarative inversion patterns
    m_how = re.search(r"\bhow does\s+([a-zA-Z0-9_-]+)\s+(.+?)\??$", clean_q, re.IGNORECASE)
    if m_how:
        subject, predicate = m_how.group(1), m_how.group(2).rstrip("?")
        return (
            f"{subject} {predicate} by coordinating distributed system workflows, "
            "optimizing memory management, and guaranteeing high availability. "
            "The architecture implements automatic failover and low-latency execution."
        )

    m_what = re.search(r"\bwhat (?:is|are)\s+([a-zA-Z0-9_-]+)\??$", clean_q, re.IGNORECASE)
    if m_what:
        subject = m_what.group(1).rstrip("?")
        return (
            f"{subject} is a core architectural component designed for high-performance data processing "
            "and service orchestration. It integrates with storage and network layers to provide "
            "reliable, scalable operations."
        )

    m_why = re.search(
        r"\bwhy (?:does|do|is)\s+([a-zA-Z0-9_-]+)\s+(.+?)\??$", clean_q, re.IGNORECASE
    )
    if m_why:
        subject, predicate = m_why.group(1), m_why.group(2).rstrip("?")
        return (
            f"{subject} {predicate} to ensure operational consistency, fault isolation, "
            "and horizontal scaling under demanding production workloads."
        )

    # Generic declarative synthesis
    q_words = [w for w in tokens(lower_q) if w not in STOP_WORDS]
    keywords = " ".join(q_words[:6])
    context_addon = f" Key systems involved include {entity_str}." if entity_str else ""
    return (
        f"In distributed infrastructure, {keywords} represents a key operational mechanism "
        f"for state management, persistence, and service reliability.{context_addon} "
        "Standard production practices ensure fault tolerance and secure multi-tenant execution."
    )


def blend_vectors(
    v_query: list[float], v_hyde: list[float], hyde_weight: float = 0.4
) -> list[float]:
    """Blend question vector and hypothetical document vector with unit L2 normalization."""
    if not v_query:
        return v_hyde
    if not v_hyde or len(v_query) != len(v_hyde):
        return v_query

    w_hyde = max(0.0, min(1.0, hyde_weight))
    w_query = 1.0 - w_hyde

    blended = [w_query * q + w_hyde * h for q, h in zip(v_query, v_hyde, strict=True)]
    norm = math.sqrt(sum(x * x for x in blended))
    if norm <= 0.0:
        return blended
    return [round(x / norm, 6) for x in blended]


def extract_pseudo_relevance_terms(text: str, original_query: str, max_terms: int = 5) -> list[str]:
    """Extract salient Pseudo-Relevance Feedback (PRF) expansion terms from hypothetical text."""
    q_tokens = set(tokens(original_query))
    doc_tokens = tokens(text)

    # Filter out tokens present in original query or stop words
    candidate_tokens = [
        t for t in doc_tokens if t not in q_tokens and t not in STOP_WORDS and len(t) >= 4
    ]

    freq = Counter(candidate_tokens)
    return [term for term, _ in freq.most_common(max_terms)]
