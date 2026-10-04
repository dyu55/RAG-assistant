"""
Late Interaction & MaxSim Token-Level Semantic Matching Engine.
Ref: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
& Santhanam et al., "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction" (2024–2026).

Implements fine-grained token-level late interaction using the MaxSim operator:
For each query token q_i, finds the maximally similar document token d_j:
MaxSim(Q, D) = (1 / |Q|) * sum_{q_i in Q} max_{d_j in D} Sim(q_i, d_j)
"""

from __future__ import annotations

from .models import Chunk
from .text import tokens


def token_similarity(q_tok: str, d_tok: str) -> float:
    """Calculate fine-grained token-level similarity between query token and document token."""
    if q_tok == d_tok:
        return 1.0

    # Common prefix root match (e.g. 'replicate' vs 'replication', 'partition' vs 'partitioning')
    prefix_len = 0
    for c1, c2 in zip(q_tok, d_tok, strict=False):
        if c1 == c2:
            prefix_len += 1
        else:
            break

    max_len = max(len(q_tok), len(d_tok))
    if prefix_len >= 4 and max_len > 0:
        ratio = prefix_len / max_len
        if ratio >= 0.65:
            return round(0.70 + 0.30 * ratio, 3)

    # Character bigram Jaccard similarity for morphological variants / typos
    if len(q_tok) >= 4 and len(d_tok) >= 4:
        q_bg = {q_tok[i : i + 2] for i in range(len(q_tok) - 1)}
        d_bg = {d_tok[i : i + 2] for i in range(len(d_tok) - 1)}
        jaccard = len(q_bg & d_bg) / len(q_bg | d_bg)
        if jaccard >= 0.65:
            return round(jaccard, 3)

    return 0.0


def compute_maxsim(query: str, document_text: str) -> tuple[float, dict[str, tuple[str, float]]]:
    """Compute ColBERT-style MaxSim token alignment score between query and document.

    Returns:
        (maxsim_score, alignments): overall score in [0.0, 1.0] and mapping
        from each query token to (best_matching_doc_token, similarity).
    """
    q_tokens = tokens(query)
    if not q_tokens:
        return 0.0, {}

    d_tokens = tokens(document_text)
    if not d_tokens:
        return 0.0, {q: ("", 0.0) for q in q_tokens}

    d_token_set = set(d_tokens)
    alignments: dict[str, tuple[str, float]] = {}
    total_score = 0.0

    for q_tok in q_tokens:
        if q_tok in d_token_set:
            alignments[q_tok] = (q_tok, 1.0)
            total_score += 1.0
            continue

        best_match = ""
        best_sim = 0.0
        for d_tok in d_token_set:
            sim = token_similarity(q_tok, d_tok)
            if sim > best_sim:
                best_sim = sim
                best_match = d_tok
                if best_sim >= 0.95:
                    break

        alignments[q_tok] = (best_match, best_sim)
        total_score += best_sim

    score = round(total_score / len(q_tokens), 4)
    return score, alignments


def late_interaction_score(query: str, chunk: Chunk) -> float:
    """Convenience evaluator returning MaxSim score for a chunk."""
    score, _ = compute_maxsim(query, chunk.text)
    return score
