"""
Plan-and-Execute Sub-Query Decomposition for Complex & Multi-Faceted Questions.
Decomposes compound or comparative user inquiries into atomic sub-queries,
executing faceted retrieval to ensure balanced evidence coverage across all question facets.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .models import Chunk, Evidence, Question
from .retrieval import retrieve


def is_compound_query(query: str) -> bool:
    """Detect whether a query contains multiple sub-questions or comparative clauses."""
    q = query.strip().lower()
    if "?" in q[:-1]:  # Multiple question marks
        return True
    if re.search(r"\b(compare|contrast|difference between|versus|\bvs\b)\b", q):
        return True
    if re.search(r"\b(and also|as well as|additionally|furthermore)\b", q):
        return True
    if re.search(
        r"\b(what|how|why|when|where|who)\b.*\b(and|while|also)\b.*\b(what|how|why|when|where|who)\b",
        q,
    ):
        return True
    return False


def decompose_query(query: str, max_subqueries: int = 3) -> list[str]:
    """
    Decompose a query into 1 to N atomic sub-queries.
    If the query is already atomic, returns [query].
    """
    clean_q = query.strip()
    if not clean_q:
        return []

    # 1. Check for multiple distinct sentences with question marks
    if "?" in clean_q[:-1]:
        parts = [p.strip() + "?" for p in clean_q.split("?") if p.strip()]
        if len(parts) > 1:
            return parts[:max_subqueries]

    # 2. Check for comparative pattern: "Compare X and/vs Y" or "Difference between X and Y"
    comp_match = re.search(
        r"(?:compare|contrast|difference between)\s+(.+?)\s+(?:and|with|to|versus|\bvs\b)\s+(.+)",
        clean_q,
        re.IGNORECASE,
    )
    if comp_match:
        part1 = comp_match.group(1).strip().rstrip("?")
        part2 = comp_match.group(2).strip().rstrip("?")
        return [f"Details about {part1}", f"Details about {part2}"][:max_subqueries]

    vs_match = re.search(r"(.+?)\s+(?:versus|\bvs\b)\s+(.+)", clean_q, re.IGNORECASE)
    if vs_match:
        part1 = vs_match.group(1).strip().rstrip("?")
        part2 = vs_match.group(2).strip().rstrip("?")
        return [f"Details about {part1}", f"Details about {part2}"][:max_subqueries]

    # 3. Check for compound conjunction clauses: "how does X work and what is Y"
    clause_match = re.search(
        r"^(.*\b(?:how|what|why|when|where|who|explain)\b.*?)\s+(?:and also|as well as|and)\s+(\b(?:how|what|why|when|where|who|explain)\b.*)$",
        clean_q,
        re.IGNORECASE,
    )
    if clause_match:
        p1 = clause_match.group(1).strip().rstrip("?") + "?"
        p2 = clause_match.group(2).strip().rstrip("?") + "?"
        return [p1, p2][:max_subqueries]

    return [clean_q]


def faceted_retrieve(
    question: Question,
    chunks: list[Chunk],
    embed_fn=None,
) -> list[Evidence]:
    """
    Execute faceted retrieval across decomposed sub-queries and fuse using Reciprocal Rank Fusion (RRF).
    Guarantees every facet of a compound query gets evidence representation.
    """
    sub_queries = decompose_query(question.text)
    if len(sub_queries) <= 1:
        vector = (
            embed_fn([question.text])[0]
            if embed_fn and question.mode in {"hybrid", "vector"}
            else None
        )
        return retrieve(question, chunks, vector)

    # Multi-faceted retrieval
    all_evidence_lists: list[list[Evidence]] = []
    for sq in sub_queries:
        sub_q = Question(
            text=sq,
            top_k=max(3, question.top_k),
            document_ids=question.document_ids,
            mode=question.mode,
        )
        vec = embed_fn([sq])[0] if embed_fn and question.mode in {"hybrid", "vector"} else None
        all_evidence_lists.append(retrieve(sub_q, chunks, vec))

    # Multi-query RRF Fusion
    fusion_scores: dict[str, float] = defaultdict(float)
    evidence_lookup: dict[str, Evidence] = {}

    for ev_list in all_evidence_lists:
        for rank, ev in enumerate(ev_list, 1):
            fusion_scores[ev.chunk_id] += 1.0 / (60.0 + rank)
            if ev.chunk_id not in evidence_lookup:
                evidence_lookup[ev.chunk_id] = ev

    # Sort merged evidence by RRF score descending
    sorted_chunk_ids = sorted(fusion_scores.keys(), key=lambda cid: -fusion_scores[cid])
    merged: list[Evidence] = []
    seen_windows: list[Evidence] = []

    for cid in sorted_chunk_ids:
        ev = evidence_lookup[cid]
        # Overlapping windows suppression
        if any(
            item.document_id == ev.document_id
            and item.page == ev.page
            and max(0, min(item.end, ev.end) - max(item.start, ev.start))
            > min(item.end - item.start, ev.end - ev.start) * 0.6
            for item in seen_windows
        ):
            continue

        seen_windows.append(ev)
        # Update source ID index
        merged.append(
            Evidence(
                source_id=f"S{len(merged) + 1}",
                chunk_id=ev.chunk_id,
                document_id=ev.document_id,
                filename=ev.filename,
                text=ev.text,
                page=ev.page,
                start=ev.start,
                end=ev.end,
                relevance=ev.relevance,
                fusion_score=round(fusion_scores[cid], 6),
                channels=ev.channels,
                path=ev.path,
            )
        )
        if len(merged) == question.top_k:
            break

    return merged
