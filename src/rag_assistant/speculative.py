"""
Speculative RAG (Drafting & Verification Engine).
Ref: Wang et al., "Speculative RAG: Enhancing Retrieval Augmented Generation
through Drafting and Verification" (2024–2026).

Implements the multi-perspective Draft-then-Verify paradigm:
1. Evidence Subset Partitioning: Divides retrieved passages into coherent subsets
   (by document source cluster or rank interleaving) to eliminate lost-in-the-middle
   position bias and context distraction.
2. Specialist Candidate Drafting: Generates candidate drafts across subsets in parallel,
   each focusing on distinct factual evidence subsets.
3. Generalist Verification & Consensus Scoring: Evaluates candidate drafts on:
   - Factual Groundedness (verbatim quote verification and textual entailment)
   - Query Alignment (intent coverage and question relevance)
   - Inter-Draft Consensus (cross-candidate mutual agreement on claims)
4. Dynamic Selection & Speculative Synthesis: Selects the dominant candidate draft or
   synthesizes complementary verified claims across diverse evidence subsets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from .models import Claim, Draft, Evidence, SpeculativeDraftCandidate, SpeculativeReport
from .text import normalize, tokens


def partition_evidence_subsets(
    evidence: list[Evidence], max_subsets: int = 3, min_per_subset: int = 1
) -> list[list[Evidence]]:
    """Partition retrieved evidence into diverse and coherent subsets."""
    if not evidence:
        return []
    if len(evidence) <= min_per_subset or max_subsets <= 1:
        return [list(evidence)]

    target_m = min(max_subsets, max(2, len(evidence) // max(1, min_per_subset)))

    # 1. Check if multiple distinct document sources exist
    doc_groups: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        doc_groups[item.document_id].append(item)

    if len(doc_groups) >= target_m:
        # Sort documents by their highest relevance item
        sorted_docs = sorted(
            doc_groups.values(),
            key=lambda group: -max((e.relevance for e in group), default=0.0),
        )
        subsets: list[list[Evidence]] = [[] for _ in range(target_m)]
        for idx, group in enumerate(sorted_docs):
            subsets[idx % target_m].extend(group)
    else:
        # 2. Interleave across subsets by retrieval rank so each subset has top-tier context
        subsets = [[] for _ in range(target_m)]
        for idx, item in enumerate(evidence):
            subsets[idx % target_m].append(item)

    # Filter out any accidentally empty subsets
    return [s for s in subsets if s]


def compute_groundedness(draft: Draft, subset: list[Evidence]) -> float:
    """Evaluate factual groundedness: ratio of claims faithful to subset evidence."""
    if not draft.claims:
        return 0.0

    sources = {e.source_id: e for e in subset}
    grounded_count = 0

    for claim in draft.claims:
        source = sources.get(claim.source_id)
        if not source:
            continue
        # Quote must exist verbatim in source text
        if normalize(claim.quote) not in normalize(source.text):
            continue
        # Words in claim must have sufficient overlap with quoted evidence
        c_words = set(tokens(claim.text))
        q_words = set(tokens(claim.quote))
        if not c_words:
            continue
        if len(c_words & q_words) / len(c_words) >= 0.45:
            grounded_count += 1

    return round(grounded_count / len(draft.claims), 3)


def compute_query_alignment(draft: Draft, query: str) -> float:
    """Evaluate query alignment: how well draft claims cover question terms."""
    q_tokens = set(tokens(query))
    if not q_tokens or not draft.claims:
        return 0.0

    claim_tokens: set[str] = set()
    for claim in draft.claims:
        claim_tokens.update(tokens(claim.text))

    overlap = len(q_tokens & claim_tokens) / len(q_tokens)
    return round(min(1.0, overlap), 3)


def compute_inter_draft_consensus(drafts: list[Draft]) -> list[float]:
    """Calculate consensus score for each draft against all other candidate drafts."""
    if len(drafts) <= 1:
        return [1.0] * len(drafts)

    draft_vocabularies = [set(tokens(" ".join(c.text for c in d.claims))) for d in drafts]

    consensus_scores: list[float] = []
    for i, vocab_i in enumerate(draft_vocabularies):
        if not vocab_i:
            consensus_scores.append(0.0)
            continue
        agreements: list[float] = []
        for j, vocab_j in enumerate(draft_vocabularies):
            if i == j:
                continue
            if not vocab_j:
                agreements.append(0.0)
                continue
            intersection = len(vocab_i & vocab_j)
            union = len(vocab_i | vocab_j)
            agreements.append(intersection / max(1, union))
        consensus_scores.append(round(sum(agreements) / max(1, len(agreements)), 3))

    return consensus_scores


def synthesize_consensus_claims(
    drafts: list[Draft],
    scores: list[float],
    evidence_map: dict[str, Evidence],
    max_claims: int = 8,
) -> Draft:
    """Synthesize complementary grounded claims across candidate drafts."""
    ranked_indices = sorted(range(len(drafts)), key=lambda i: -scores[i])
    selected_claims: list[Claim] = []
    seen_texts: list[set[str]] = []

    for idx in ranked_indices:
        draft = drafts[idx]
        for claim in draft.claims:
            ev = evidence_map.get(claim.source_id)
            if not ev or normalize(claim.quote) not in normalize(ev.text):
                continue
            c_words = set(tokens(claim.text))
            if not c_words:
                continue
            # Deduplicate if heavily overlapping with an already chosen claim
            is_dup = any(len(c_words & prev) / len(c_words) >= 0.70 for prev in seen_texts)
            if not is_dup:
                selected_claims.append(claim)
                seen_texts.append(c_words)
            if len(selected_claims) >= max_claims:
                break
        if len(selected_claims) >= max_claims:
            break

    return Draft(claims=selected_claims)


def execute_speculative_rag(
    query: str,
    evidence: list[Evidence],
    drafter_fn: Callable[[str, list[Evidence]], Draft],
    max_subsets: int = 3,
    synthesis_threshold: float = 0.60,
) -> tuple[SpeculativeReport, Draft]:
    """Execute Speculative RAG: partition, parallel draft, verify, and select/synthesize."""
    if not evidence:
        report = SpeculativeReport(
            enabled=True,
            num_subsets=0,
            subsets=[],
            drafts=[],
            selected_index=0,
            selection_strategy="single_pass",
            consensus_score=0.0,
            rationale="No retrieved evidence available.",
        )
        return report, Draft(claims=[])

    if len(evidence) < 2:
        single_draft = drafter_fn(query, evidence)
        src_ids = [e.source_id for e in evidence]
        g_score = compute_groundedness(single_draft, evidence)
        a_score = compute_query_alignment(single_draft, query)
        cand = SpeculativeDraftCandidate(
            index=0,
            source_ids=src_ids,
            claim_count=len(single_draft.claims),
            groundedness=g_score,
            query_alignment=a_score,
            consensus_score=1.0,
            total_score=round(0.45 * g_score + 0.35 * a_score + 0.20, 3),
            summary=single_draft.claims[0].text[:80] if single_draft.claims else "Empty",
        )
        report = SpeculativeReport(
            enabled=True,
            num_subsets=1,
            subsets=[src_ids],
            drafts=[cand],
            selected_index=0,
            selection_strategy="single_pass",
            consensus_score=1.0,
            rationale="Single evidence passage; executed direct drafting.",
        )
        return report, single_draft

    subsets = partition_evidence_subsets(evidence, max_subsets=max_subsets)

    # 1. Parallel drafting across subsets
    with ThreadPoolExecutor(max_workers=min(len(subsets), 4)) as executor:
        drafts = list(executor.map(lambda sub: drafter_fn(query, sub), subsets))

    # 2. Candidate evaluation & verification scoring
    groundedness = [compute_groundedness(d, sub) for d, sub in zip(drafts, subsets, strict=True)]
    query_align = [compute_query_alignment(d, query) for d in drafts]
    consensus = compute_inter_draft_consensus(drafts)

    total_scores = [
        round(0.45 * g + 0.35 * a + 0.20 * c, 3)
        for g, a, c in zip(groundedness, query_align, consensus, strict=True)
    ]

    candidates: list[SpeculativeDraftCandidate] = []
    for idx, (sub, d) in enumerate(zip(subsets, drafts, strict=True)):
        summary_txt = d.claims[0].text[:80] if d.claims else "No claims"
        candidates.append(
            SpeculativeDraftCandidate(
                index=idx,
                source_ids=[e.source_id for e in sub],
                claim_count=len(d.claims),
                groundedness=groundedness[idx],
                query_alignment=query_align[idx],
                consensus_score=consensus[idx],
                total_score=total_scores[idx],
                summary=summary_txt,
            )
        )

    all_evidence_map = {e.source_id: e for e in evidence}
    avg_consensus = round(sum(consensus) / max(1, len(consensus)), 3)

    # 3. Strategy decision: Consensus synthesis vs Best candidate selection
    qualified = [
        i
        for i in range(len(drafts))
        if groundedness[i] >= 0.70 and query_align[i] >= 0.15 and len(drafts[i].claims) > 0
    ]

    if len(qualified) >= 2:
        synthesized = synthesize_consensus_claims(drafts, total_scores, all_evidence_map)
        max_single_claims = max((len(drafts[i].claims) for i in qualified), default=0)
        if len(synthesized.claims) > max_single_claims:
            report = SpeculativeReport(
                enabled=True,
                num_subsets=len(subsets),
                subsets=[[e.source_id for e in s] for s in subsets],
                drafts=candidates,
                selected_index=-1,
                selection_strategy="consensus_synthesis",
                consensus_score=avg_consensus,
                rationale=(
                    f"Synthesized {len(synthesized.claims)} verified consensus claims across "
                    f"{len(qualified)} high-fidelity candidate drafts."
                ),
            )
            return report, synthesized

    # Best candidate selection
    best_idx = max(range(len(drafts)), key=lambda i: total_scores[i])
    if drafts[best_idx].claims and groundedness[best_idx] > 0:
        report = SpeculativeReport(
            enabled=True,
            num_subsets=len(subsets),
            subsets=[[e.source_id for e in s] for s in subsets],
            drafts=candidates,
            selected_index=best_idx,
            selection_strategy="best_candidate",
            consensus_score=avg_consensus,
            rationale=(
                f"Selected candidate {best_idx} with top verification score {total_scores[best_idx]} "
                f"(Groundedness={groundedness[best_idx]}, Alignment={query_align[best_idx]})."
            ),
        )
        return report, drafts[best_idx]

    # Fallback to single pass if all candidate drafts failed verification
    fallback_draft = drafter_fn(query, evidence)
    report = SpeculativeReport(
        enabled=True,
        num_subsets=len(subsets),
        subsets=[[e.source_id for e in s] for s in subsets],
        drafts=candidates,
        selected_index=-1,
        selection_strategy="fallback_single_pass",
        consensus_score=avg_consensus,
        rationale="Candidate drafts failed verification; safely fell back to full context drafting.",
    )
    return report, fallback_draft
