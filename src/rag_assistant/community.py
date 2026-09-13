from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from .models import Chunk, CommunityReport, Evidence, Question
from .text import tokens


def build_entity_graph(
    chunks: list[Chunk],
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]], dict[str, Chunk]]:
    """Build entity-to-chunks map, entity-entity weighted co-occurrence graph, and chunk lookup."""
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    neighbors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    chunk_lookup: dict[str, Chunk] = {c.id: c for c in chunks}

    for chunk in chunks:
        norm_entities = sorted({e.strip().lower() for e in chunk.entities if len(e.strip()) > 1})
        for entity in norm_entities:
            entity_to_chunks[entity].add(chunk.id)
        for left, right in combinations(norm_entities, 2):
            neighbors[left][right] += 1
            neighbors[right][left] += 1

    return entity_to_chunks, neighbors, chunk_lookup


def detect_communities(chunks: list[Chunk]) -> list[CommunityReport]:
    """Lightweight GraphRAG community detection via Label Propagation over entity co-occurrence."""
    entity_to_chunks, neighbors, chunk_lookup = build_entity_graph(chunks)
    if not neighbors:
        return []

    entities = sorted(neighbors.keys())
    labels = {e: e for e in entities}

    # Label propagation iterations
    for _ in range(5):
        changed = False
        for entity in entities:
            nbr_weights: dict[str, int] = defaultdict(int)
            for nbr, weight in neighbors[entity].items():
                nbr_weights[labels[nbr]] += weight
            if nbr_weights:
                best_label = max(nbr_weights.items(), key=lambda item: (item[1], item[0]))[0]
                if labels[entity] != best_label:
                    labels[entity] = best_label
                    changed = True
        if not changed:
            break

    # Group entities by assigned community label
    clusters: dict[str, list[str]] = defaultdict(list)
    for entity, label in labels.items():
        clusters[label].append(entity)

    reports: list[CommunityReport] = []
    for idx, (_label, cluster_entities) in enumerate(
        sorted(clusters.items(), key=lambda item: -len(item[1]))
    ):
        # Calculate degree centrality within community
        internal_degree = {
            e: sum(neighbors[e].get(other, 0) for other in cluster_entities if other != e)
            for e in cluster_entities
        }
        sorted_entities = sorted(cluster_entities, key=lambda e: (-internal_degree[e], e))
        hub_entities = sorted_entities[:4]

        # Synthesize a descriptive title from top hub entities
        if len(hub_entities) >= 2:
            title = f"{hub_entities[0].title()} & {hub_entities[1].title()} Subsystem"
        elif hub_entities:
            title = f"{hub_entities[0].title()} Cluster"
        else:
            title = f"Community {idx + 1}"

        # Collect unique chunk IDs
        comm_chunk_ids = sorted({cid for e in cluster_entities for cid in entity_to_chunks[e]})

        # Extract thematic summary sentences containing hub entities
        candidate_sentences = []
        for cid in comm_chunk_ids:
            chunk = chunk_lookup[cid]
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
            for sent in sentences:
                sent_clean = sent.strip()
                if 20 <= len(sent_clean) <= 250:
                    matched_hubs = sum(1 for hub in hub_entities if hub in sent_clean.lower())
                    if matched_hubs > 0:
                        candidate_sentences.append((matched_hubs, len(sent_clean), sent_clean))

        candidate_sentences.sort(key=lambda s: (-s[0], s[1]))
        seen_sents: set[str] = set()
        chosen: list[str] = []
        for _, _, sent in candidate_sentences:
            if sent not in seen_sents:
                seen_sents.add(sent)
                chosen.append(sent)
                if len(chosen) >= 2:
                    break

        if chosen:
            summary = " ".join(chosen)
        else:
            summary = (
                f"Knowledge community centered around {', '.join(h.title() for h in hub_entities)}."
            )

        weight = sum(internal_degree.values()) // 2 + len(comm_chunk_ids)
        reports.append(
            CommunityReport(
                id=f"comm_{idx + 1}",
                title=title,
                hub_entities=[h.title() for h in hub_entities],
                entities=sorted(e.title() for e in cluster_entities),
                chunk_ids=comm_chunk_ids,
                summary=summary,
                weight=weight,
            )
        )

    return reports


GLOBAL_QUERY_PATTERNS = (
    r"\b(summarize (all|the entire|the overall)|overview of (all|the)|main themes|all components)\b",
    r"\b(key systems|big picture|across all documents|high-level topics|ecosystem|architecture summary)\b",
    r"\b(what are all the|how do all|global summary)\b",
)


def is_global_query(query: str) -> bool:
    """Detect whether a query requests global panoramic knowledge graph community synthesis."""
    q_clean = query.strip().lower()
    return any(re.search(pat, q_clean) for pat in GLOBAL_QUERY_PATTERNS)


def retrieve_communities(
    query: str, communities: list[CommunityReport], top_k: int = 3
) -> list[CommunityReport]:
    """Rank and retrieve the most relevant community reports for a given query."""
    if not communities:
        return []

    q_tokens = set(tokens(query))
    scored: list[tuple[float, CommunityReport]] = []

    for comm in communities:
        title_tokens = set(tokens(comm.title))
        hub_tokens = {t for h in comm.hub_entities for t in tokens(h)}
        summary_tokens = set(tokens(comm.summary))

        score = (
            3.0 * len(q_tokens & title_tokens)
            + 2.0 * len(q_tokens & hub_tokens)
            + 1.0 * len(q_tokens & summary_tokens)
        )
        scored.append((score, comm))

    scored.sort(key=lambda item: (-item[0], -item[1].weight, item[1].id))
    # If query has matches, return matched communities; otherwise return top weighted communities
    top_matches = [comm for score, comm in scored if score > 0]
    if top_matches:
        return top_matches[:top_k]
    return [comm for _, comm in scored[:top_k]]


def global_community_retrieve(
    question: Question, chunks: list[Chunk], top_k: int | None = None
) -> list[Evidence]:
    """Perform Global GraphRAG retrieval: detects communities and gathers representative evidence

    from each relevant community with contextual summary windows.
    """
    communities = detect_communities(chunks)
    if not communities:
        return []

    k = top_k or question.top_k
    relevant_communities = retrieve_communities(question.text, communities, top_k=min(k, 4))
    chunk_lookup = {c.id: c for c in chunks}

    selected_evidence: list[Evidence] = []
    seen_chunks: set[str] = set()

    for comm in relevant_communities:
        # Pick the most central chunk in this community that has not been selected
        for cid in comm.chunk_ids:
            if cid not in seen_chunks and cid in chunk_lookup:
                seen_chunks.add(cid)
                chunk = chunk_lookup[cid]
                context_win = f"[Community: {comm.title}] {comm.summary}\n\n{chunk.text}"
                selected_evidence.append(
                    Evidence(
                        source_id=f"S{len(selected_evidence) + 1}",
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        text=chunk.text,
                        page=chunk.page,
                        start=chunk.start,
                        end=chunk.end,
                        relevance=round(min(1.0, 0.80 + 0.05 * min(len(comm.hub_entities), 3)), 4),
                        rerank_score=round(
                            min(1.0, 0.80 + 0.05 * min(len(comm.hub_entities), 3)), 4
                        ),
                        fusion_score=round(1.0 / (60.0 + len(selected_evidence) + 1), 6),
                        channels=["graph", "community"],
                        path=comm.hub_entities[:3],
                        context_window=context_win,
                    )
                )
                break
        if len(selected_evidence) >= k:
            break

    return selected_evidence
