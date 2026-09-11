"""
Corrective RAG (CRAG) & Self-RAG Inference-Time Correction Guardrail.
Provides:
1. Pre-generation Retrieval Gate: Detects weak retrieval and applies corrective query expansion.
2. Post-generation Faithfulness Gate: Rescues ungrounded or hallucinated drafts via verified extractive synthesis.
3. Audit-Grade Traceability: Records all correction actions in Answer metadata.
"""

from __future__ import annotations

import re

from .models import Draft, Evidence
from .providers import extractive_draft
from .text import tokens


def is_retrieval_insufficient(evidence: list[Evidence], threshold: float = 0.20) -> bool:
    """Check if retrieved evidence is empty or has poor relevance scores."""
    if not evidence:
        return True
    return max(e.relevance for e in evidence) < threshold


def correct_query_for_fallback(query: str) -> str:
    """
    Reformulate user query by stripping conversational pleasantries,
    interrogative filler, and punctuation to broaden lexical recall.
    """
    clean_q = query.strip()
    # Strip common conversational prefixes
    clean_q = re.sub(
        r"^(please|could you please|can you tell me|i want to know|tell me about|explain to me)\s+",
        "",
        clean_q,
        flags=re.IGNORECASE,
    )
    # Strip trailing punctuation
    clean_q = clean_q.rstrip("?!. ")

    # Extract substantive terms
    terms = [t for t in tokens(clean_q) if len(t) > 2]
    if terms:
        return " ".join(terms)
    return clean_q


def rescue_unsupported_draft(
    question_text: str,
    draft: Draft,
    evidence: list[Evidence],
    verify_fn,
) -> tuple[Draft, bool, str]:
    """
    Inference-time critic (Self-RAG):
    When an initial LLM draft fails strict quote or support verification,
    attempt an extractive fallback anchored strictly in verbatim retrieved passages.
    Returns (resolved_draft, is_rescued, reason).
    """
    # First check if the original draft is already verified
    supported, reason = verify_fn(draft, evidence)
    if supported:
        return draft, False, reason

    # Attempt extractive self-correction
    if not evidence:
        return draft, False, reason

    fallback = extractive_draft(question_text, evidence)
    fb_supported, fb_reason = verify_fn(fallback, evidence)

    if fb_supported and fallback.claims:
        return fallback, True, "Rescued unverified draft via verified extractive synthesis."

    # Both failed verification
    return draft, False, reason
