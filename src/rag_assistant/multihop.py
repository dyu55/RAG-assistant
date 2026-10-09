"""
Iterative Multi-Hop Agentic Retrieval & Bridge Entity Chaining Engine.
Ref: Khattab et al., "Demonstrate-Search-Predict (DSPy)" / "Baleen: Multi-Hop
Reasoning over Dense Retrieval" (Stanford, 2023–2025);
"Iterative Retrieval-Augmented Generation" (ACL 2025–2026).

Implements iterative multi-step reasoning across complex relational queries:
1. Multi-Hop Intent Detection: Detects dependent relational and anaphoric clauses
   (e.g., "what database does X use and how does it replicate?", "who created Y and why?").
2. Hop 1 (Initial Grounding & Bridge Entity Extraction): Identifies intermediate entities
   and concepts that surface in Hop 1 evidence but were absent in the user query.
3. Adaptive Follow-Up Query Synthesis: Replaces vague relational pronouns ("it", "that", "its")
   with grounded bridge entities to formulate targeted follow-up queries.
4. Hop 2 (Chained Retrieval & Multi-Hop Fusion): Gathers missing relational evidence,
   fuses cross-hop candidate pools via reciprocal deduplication, and records multi-hop trace provenance.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from .models import Chunk, Evidence, MultiHopTrace, Question
from .text import tokens

MULTIHOP_PATTERNS = [
    r"\b(?:and|while|also)\s+(?:how|what|why|where|who)\s+(?:does|is|are|do|can)\s+(?:it|this|that|these|the same|its)\b",
    r"\b(?:which|what)\s+(?:\w+\s+){1,4}(?:uses?|created|built|designed|developed|powers?)\s+.*?\s+and\s+(?:how|what|why|who)\b",
    r"\b(?:who|what)\s+(?:is|was)\s+(?:the\s+)?(?:\w+\s+)?(?:behind|founder|creator|author)\s+of\s+the\s+(?:\w+\s+)?(?:that|which)\b",
    r"\b(?:its|their)\s+(?:replication|consistency|consensus|architecture|failover|algorithm|strategy|protocol|founder|author|creator)\b",
    r"\b(?:how does it|how is it|what does it|why does it)\b",
]


def detect_multihop_need(query: str) -> bool:
    """Detect whether a query exhibits anaphoric dependency or relational multi-hop structure."""
    q = query.strip().lower()
    if len(q) < 18:
        return False
    return any(re.search(p, q) for p in MULTIHOP_PATTERNS)


def extract_bridge_entities(
    query: str, evidence: list[Evidence], chunks: list[Chunk] | None = None
) -> list[str]:
    """Extract candidate bridge entities appearing in Hop 1 evidence that resolve intermediate relations."""
    if not evidence:
        return []

    q_words = set(tokens(query.lower()))
    candidate_counts: Counter[str] = Counter()

    # 1. Harvest explicit entity tags from evidence chunks
    chunk_map = {c.id: c for c in chunks} if chunks else {}
    for ev in evidence:
        c = chunk_map.get(ev.chunk_id)
        if c and c.entities:
            for ent in c.entities:
                norm_ent = ent.strip()
                if norm_ent and not (set(tokens(norm_ent.lower())) <= q_words):
                    candidate_counts[norm_ent] += 3

    # 2. Extract capitalized proper nouns and technical terms from evidence text
    for ev in evidence:
        matches = re.findall(
            r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b",
            ev.text,
        )
        for m in matches:
            cleaned = m.strip()
            # Ignore standard sentence starters
            if cleaned.lower() in {
                "the",
                "this",
                "that",
                "these",
                "section",
                "chapter",
                "note",
                "table",
                "figure",
            }:
                continue
            if len(cleaned) >= 3 and not (set(tokens(cleaned.lower())) <= q_words):
                candidate_counts[cleaned] += 1

    # Filter candidates with at least 1 frequency, returning top bridge entities
    ranked = [ent for ent, count in candidate_counts.most_common(5) if count >= 1]
    return ranked[:3]


def synthesize_followup_query(query: str, bridge_entities: list[str]) -> str:
    """Synthesize a grounded follow-up query replacing vague references with the bridge entity."""
    if not bridge_entities:
        return query

    primary_bridge = bridge_entities[0]

    # Try splitting on conjunction clauses (e.g., "what store does X use and how does it handle failover?")
    clause_match = re.search(
        r"(?:and|while|also)\s+(?:how|what|why|where|who)\s+(?:does|is|are|do|can)\s+(?:it|this|that|these|the same|its)\s+(.*)",
        query,
        re.IGNORECASE,
    )
    if clause_match:
        tail = clause_match.group(1).strip().rstrip("?")
        # Reconstruct follow-up question directly addressing the second hop
        return f"How does {primary_bridge} {tail}?"

    # Try pronoun substitution: "its replication" -> "Redis replication"
    replaced = re.sub(
        r"\b(?:it|this|that|its|their|the database|the store|the system)\b",
        primary_bridge,
        query,
        flags=re.IGNORECASE,
    )
    if replaced.strip().lower() != query.strip().lower():
        return replaced.strip()

    # Fallback: concatenate primary bridge entity with non-stopword query tokens
    return f"{primary_bridge} {query}"


def execute_multihop_search(
    question: Question,
    chunks: list[Chunk],
    retrieve_fn: Callable[[Question], list[Evidence]],
    max_hops: int = 2,
) -> tuple[list[Evidence], MultiHopTrace]:
    """Execute iterative multi-hop retrieval loop and fuse cross-hop evidence."""
    # Hop 1: Initial grounding retrieval
    hop1_evidence = retrieve_fn(question)
    if not hop1_evidence or max_hops <= 1 or not detect_multihop_need(question.text):
        trace = MultiHopTrace(
            hops_executed=1,
            bridge_entities=[],
            sub_queries=[question.text],
            resolved=True,
            reasoning="Single-hop retrieval fulfilled query scope.",
        )
        return hop1_evidence, trace

    # Extract bridge entities from Hop 1
    bridges = extract_bridge_entities(question.text, hop1_evidence, chunks)
    if not bridges:
        trace = MultiHopTrace(
            hops_executed=1,
            bridge_entities=[],
            sub_queries=[question.text],
            resolved=True,
            reasoning="No intermediate bridge entities identified.",
        )
        return hop1_evidence, trace

    # Synthesize follow-up query for Hop 2
    followup_q = synthesize_followup_query(question.text, bridges)
    if followup_q.strip().lower() == question.text.strip().lower():
        trace = MultiHopTrace(
            hops_executed=1,
            bridge_entities=bridges,
            sub_queries=[question.text],
            resolved=True,
            reasoning="Follow-up query was identical to root query.",
        )
        return hop1_evidence, trace

    # Hop 2: Chained retrieval for missing relational context
    hop2_question = question.model_copy(update={"text": followup_q})
    hop2_evidence = retrieve_fn(hop2_question)

    # Fuse evidence from Hop 1 and Hop 2
    seen_chunk_ids: set[str] = set()
    fused_evidence: list[Evidence] = []

    # Tag provenance
    for e in hop1_evidence:
        if e.chunk_id not in seen_chunk_ids:
            tagged = e.model_copy(update={"path": [*e.path, "multihop:hop_1"]})
            fused_evidence.append(tagged)
            seen_chunk_ids.add(e.chunk_id)

    for e in hop2_evidence:
        if e.chunk_id not in seen_chunk_ids:
            tagged = e.model_copy(update={"path": [*e.path, "multihop:hop_2"]})
            fused_evidence.append(tagged)
            seen_chunk_ids.add(e.chunk_id)

    # Sort by relevance descending, limiting to target top_k (with slight headroom)
    fused_evidence.sort(key=lambda ev: (-ev.relevance, -ev.rerank_score))
    target_k = max(question.top_k, min(len(fused_evidence), question.top_k + 2))
    final_evidence = fused_evidence[:target_k]

    # Re-normalize source IDs (S1, S2, ...)
    for idx, ev in enumerate(final_evidence, 1):
        ev.source_id = f"S{idx}"

    trace = MultiHopTrace(
        hops_executed=2,
        bridge_entities=bridges,
        sub_queries=[question.text, followup_q],
        resolved=True,
        reasoning=(
            f"Successfully executed 2-hop iterative retrieval: bridged via '{bridges[0]}' "
            f"to follow-up query '{followup_q}'."
        ),
    )

    return final_evidence, trace
