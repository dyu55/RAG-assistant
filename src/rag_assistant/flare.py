"""
FLARE: Forward-Looking Active REtrieval & Dynamic Lookahead Grounding.
Ref: Jiang et al., "Active Retrieval Augmented Generation" (EMNLP 2023 / 2024–2026).

Implements the active retrieval paradigm for long-form generation:
1. Lookahead Uncertainty Detection: Continuously monitors generated candidate claims/sentences.
   Detects whether a statement lacks grounded support, references missing citations, or has
   low token confidence/overlap with the current retrieved evidence pool.
2. Active Query Synthesis: Automatically extracts high-information semantic search queries
   from lookahead sentences by masking stopwords, transitions, and conversational artifacts.
3. Iterative Dynamic Retrieval: Pauses generation to retrieve targeted supplementary evidence
   specifically addressing the uncertain lookahead statement.
4. Factual Regrounding & Alignment: Identifies the most relevant sentence quote in the newly
   retrieved passages and anchors the lookahead claim to verifiable evidence.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import Claim, Draft, Evidence, FlareStep, FlareTrace, Question
from .text import normalize, tokens

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "by",
        "with",
        "from",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "also",
        "such",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
    }
)

TRANSITION_PREFIX_PATTERN = re.compile(
    r"^(furthermore|additionally|moreover|in addition|therefore|consequently|specifically|in summary|subsequently|finally|meanwhile|next|thus|hence)[,\s]+",
    re.IGNORECASE,
)

CITATION_PATTERN = re.compile(r"\[S\d+\]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")


def detect_lookahead_uncertainty(
    claim: Claim,
    evidence_map: dict[str, Evidence],
    threshold: float = 0.60,
) -> bool:
    """Detect whether a lookahead claim exhibits factual uncertainty or ungrounded attribution.

    Returns True if:
    - The cited source ID does not exist in the retrieved evidence pool.
    - The claim's quote is empty or not found verbatim in the cited source text.
    - The lexical overlap between the claim text and cited passage is below threshold.
    """
    if not claim.text.strip():
        return False

    source = evidence_map.get(claim.source_id)
    if not source:
        return True

    if not claim.quote.strip():
        return True

    # Quote must be present in the source passage
    if normalize(claim.quote) not in normalize(source.text):
        return True

    # Check token overlap confidence
    c_tokens = set(tokens(claim.text))
    if not c_tokens:
        return False

    s_tokens = set(tokens(source.text))
    overlap_score = len(c_tokens & s_tokens) / max(1, len(c_tokens))

    return overlap_score < threshold


def synthesize_active_query(lookahead_text: str) -> str:
    """Synthesize a focused retrieval query from an uncertain lookahead sentence.

    Strips citations, leading discourse markers, and stopwords to focus on core semantic entities.
    """
    cleaned = CITATION_PATTERN.sub("", lookahead_text).strip()
    cleaned = TRANSITION_PREFIX_PATTERN.sub("", cleaned).strip()

    # Extract informative tokens
    toks = tokens(cleaned)
    filtered = [t for t in toks if t.lower() not in STOPWORDS and len(t) > 1]

    if len(filtered) >= 2:
        return " ".join(filtered)

    # Fallback to cleaned sentence if token filtering was overly aggressive
    return cleaned if cleaned else lookahead_text.strip()


def reground_claim_with_evidence(
    claim: Claim,
    candidate_evidence: list[Evidence],
    min_overlap: float = 0.25,
) -> tuple[Claim | None, Evidence | None]:
    """Search candidate evidence passages to find factual support and reground the lookahead claim."""
    claim_norm = normalize(claim.text)
    claim_toks = set(tokens(claim.text))

    best_match_ev: Evidence | None = None
    best_quote: str = ""
    best_score: float = 0.0

    for ev in candidate_evidence:
        sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(ev.text) if s.strip()]
        for sentence in sentences:
            sent_norm = normalize(sentence)

            # Substring containment is strong signal
            if claim_norm in sent_norm or sent_norm in claim_norm:
                score = 1.0
            else:
                sent_toks = set(tokens(sentence))
                if not sent_toks or not claim_toks:
                    continue
                score = len(claim_toks & sent_toks) / max(1, len(claim_toks))

            if score > best_score and score >= min_overlap:
                best_score = score
                best_quote = sentence
                best_match_ev = ev

    if best_match_ev and best_quote:
        regrounded = Claim(
            source_id=best_match_ev.source_id,
            text=claim.text,
            quote=best_quote,
        )
        return regrounded, best_match_ev

    return None, None


def execute_flare_active_retrieval(
    question: str,
    draft: Draft,
    evidence: list[Evidence],
    retrieve_fn: Callable[[Question], list[Evidence]],
    max_lookaheads: int = 2,
    confidence_threshold: float = 0.60,
) -> tuple[Draft, list[Evidence], FlareTrace]:
    """Execute FLARE (Forward-Looking Active REtrieval) over draft claims.

    Identifies uncertain or ungrounded claims in the generated draft, triggers forward-looking
    targeted active retrieval queries, and updates claims with grounded evidence citations.
    """
    if not draft.claims or max_lookaheads <= 0:
        trace = FlareTrace(
            enabled=True,
            iterations=0,
            reasoning="No claims to evaluate or lookaheads disabled",
        )
        return draft, evidence, trace

    evidence_map = {e.source_id: e for e in evidence}
    existing_chunk_ids = {e.chunk_id for e in evidence}
    active_evidence = list(evidence)
    resolved_claims = list(draft.claims)

    steps: list[FlareStep] = []
    lookahead_queries: list[str] = []
    supplementary_ids: list[str] = []
    lookaheads_performed = 0
    regrounded_count = 0

    for idx, claim in enumerate(draft.claims):
        if lookaheads_performed >= max_lookaheads:
            break

        is_uncertain = detect_lookahead_uncertainty(claim, evidence_map, confidence_threshold)
        if not is_uncertain:
            continue

        lookahead_query = synthesize_active_query(claim.text)
        lookahead_queries.append(lookahead_query)
        lookaheads_performed += 1

        try:
            candidates = retrieve_fn(Question(text=lookahead_query, mode="hybrid", limit=3))
        except Exception:
            candidates = []

        # Filter and assign clean source IDs to new candidates
        usable_candidates: list[Evidence] = []
        for c in candidates:
            if c.chunk_id in existing_chunk_ids:
                # Reuse existing source item
                matching_ev = next((e for e in active_evidence if e.chunk_id == c.chunk_id), c)
                usable_candidates.append(matching_ev)
            else:
                new_source_id = f"S{len(active_evidence) + 1}"
                updated_c = c.model_copy(update={"source_id": new_source_id})
                usable_candidates.append(updated_c)

        regrounded_claim, matched_ev = reground_claim_with_evidence(claim, usable_candidates)

        if regrounded_claim and matched_ev:
            resolved_claims[idx] = regrounded_claim
            if matched_ev.chunk_id not in existing_chunk_ids:
                active_evidence.append(matched_ev)
                existing_chunk_ids.add(matched_ev.chunk_id)
                evidence_map[matched_ev.source_id] = matched_ev
                supplementary_ids.append(matched_ev.source_id)

            regrounded_count += 1
            steps.append(
                FlareStep(
                    lookahead_text=claim.text,
                    query_generated=lookahead_query,
                    retrieved_source_id=matched_ev.source_id,
                    confidence_score=round(matched_ev.relevance, 3),
                    grounded=True,
                )
            )
        else:
            steps.append(
                FlareStep(
                    lookahead_text=claim.text,
                    query_generated=lookahead_query,
                    retrieved_source_id=None,
                    confidence_score=0.0,
                    grounded=False,
                )
            )

    trace = FlareTrace(
        enabled=True,
        iterations=lookaheads_performed,
        active_retrievals=steps,
        lookahead_queries=lookahead_queries,
        supplementary_evidence_ids=supplementary_ids,
        regrounded_claims_count=regrounded_count,
        reasoning=(
            f"Executed {lookaheads_performed} FLARE lookahead(s); successfully grounded {regrounded_count} statement(s)."
            if lookaheads_performed > 0
            else "All generated draft claims satisfied confidence criteria without active retrieval."
        ),
    )

    return Draft(claims=resolved_claims), active_evidence, trace
