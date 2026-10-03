"""
Self-RAG (Self-Reflective Retrieval-Augmented Generation) & Reflection Critic Engine.
Ref: Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (2024–2026).

Implements fine-grained inference-time reflection tokens:
1. [Retrieve] / [NoRetrieve]: Evaluates whether query requires external knowledge retrieval.
2. [IsRel] / [NotRel]: Critiques relevance of individual retrieved evidence passages.
3. [IsSup:fully] / [IsSup:partial] / [IsSup:none]: Claim-level factual grounding and entailment critique.
4. [IsUse:1..5]: Answer utility assessment for user question resolution.
5. Self-Healing Closed Loop: Automatically prunes unsupported/hallucinated claims while preserving valid ones.
"""

from __future__ import annotations

import re

from .models import Claim, Draft, Evidence, ReflectionReport
from .text import normalize, tokens

GREETING_PATTERNS = (
    r"^(hi|hello|hey|good morning|good afternoon|good evening|thanks|thank you|bye|goodbye)\b",
)


def critique_retrieval_need(query: str) -> str:
    """Evaluate whether the user query requires knowledge retrieval or is purely conversational."""
    cleaned = query.strip().lower()
    if any(re.match(p, cleaned) for p in GREETING_PATTERNS) and len(cleaned.split()) <= 3:
        return "[NoRetrieve]"
    return "[Retrieve]"


def critique_passage_relevance(passage: Evidence, query: str) -> str:
    """Evaluate whether a retrieved passage is relevant to the question."""
    q_tokens = set(tokens(query))
    if not q_tokens:
        return "[NotRel]"

    p_tokens = set(tokens(passage.text))
    overlap = len(q_tokens & p_tokens) / len(q_tokens)

    # Relevant if sufficient token overlap or high retriever relevance score
    if overlap >= 0.15 or passage.relevance >= 0.40:
        return "[IsRel]"
    return "[NotRel]"


def critique_claim_support(
    claim: Claim, evidence_map: dict[str, Evidence]
) -> tuple[str, float, str]:
    """Critique factual entailment of an individual claim against its cited passage.

    Returns (reflection_token, overlap_score, justification).
    """
    source = evidence_map.get(claim.source_id)
    if not source:
        return "[IsSup:none]", 0.0, "Cited source ID not found in retrieved evidence"

    # Quote must be verbatim substring in source passage
    if normalize(claim.quote) not in normalize(source.text):
        return "[IsSup:none]", 0.0, "Quotation is hallucinated; not found in cited source text"

    c_words = set(tokens(claim.text))
    q_words = set(tokens(claim.quote))
    if not c_words:
        return "[IsSup:none]", 0.0, "Claim text has no substantive terms"

    overlap = len(c_words & q_words) / len(c_words)

    if overlap >= 0.70:
        return "[IsSup:fully]", round(overlap, 3), "Claim is fully substantiated by quotation"
    elif overlap >= 0.45:
        return "[IsSup:partial]", round(overlap, 3), "Claim is partially substantiated by quotation"
    else:
        return "[IsSup:none]", round(overlap, 3), "Insufficient textual support in quotation"


def critique_answer_utility(query: str, answer_text: str, claims: list[Claim], status: str) -> int:
    """Assess response utility and helpfulness on a 1-5 scale ([IsUse:1..5])."""
    if status == "abstained":
        return 1

    if not claims or not answer_text.strip():
        return 1

    q_tokens = set(tokens(query))
    a_tokens = set(tokens(answer_text))
    overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))

    # Scale 1-5 based on grounding count and query coverage
    base_score = 3
    if overlap >= 0.50:
        base_score += 1
    if len(claims) >= 2:
        base_score += 1

    return min(5, max(1, base_score))


def execute_self_reflection(
    query: str,
    evidence: list[Evidence],
    draft: Draft,
    status: str,
) -> tuple[ReflectionReport, Draft, bool]:
    """Execute complete Self-RAG reflection cycle and apply self-healing claim pruning if necessary."""
    evidence_map = {e.source_id: e for e in evidence}
    retrieve_token = critique_retrieval_need(query)

    # Passage relevance critique
    passage_tokens = {e.source_id: critique_passage_relevance(e, query) for e in evidence}

    # Claim-level support critique
    claim_evals: list[dict[str, str]] = []
    valid_indices: list[int] = []

    for idx, claim in enumerate(draft.claims):
        token, score, justification = critique_claim_support(claim, evidence_map)
        claim_evals.append(
            {
                "claim_idx": str(idx),
                "source_id": claim.source_id,
                "token": token,
                "score": str(score),
                "reason": justification,
            }
        )
        if token in {"[IsSup:fully]", "[IsSup:partial]"}:
            valid_indices.append(idx)

    was_healed = False
    resolved_draft = draft

    # Self-healing: if some claims are supported but others are unsupported, prune unsupported claims
    if 0 < len(valid_indices) < len(draft.claims):
        pruned_claims = [draft.claims[i] for i in valid_indices]
        resolved_draft = Draft(claims=pruned_claims)
        was_healed = True

    final_claims = resolved_draft.claims if status == "supported" or was_healed else []
    utility = critique_answer_utility(
        query,
        " ".join(c.text for c in final_claims),
        final_claims,
        "supported" if final_claims else status,
    )

    verdict = (
        "supported"
        if final_claims
        and all(
            e["token"] != "[IsSup:none]" for i, e in enumerate(claim_evals) if i in valid_indices
        )
        else "abstained"
    )

    report = ReflectionReport(
        retrieve_decision=retrieve_token,
        passage_relevance=passage_tokens,
        claim_support=claim_evals,
        utility_score=utility,
        verdict=verdict,
        healed=was_healed,
    )

    return report, resolved_draft, was_healed
