"""
HippoRAG: Neurobiologically Inspired Associative Memory & Personalized PageRank Retrieval.
Ref: Bernal et al., "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models" (ICML 2025/2026).

Leverages a Knowledge Graph as the hippocampus and Personalized PageRank (PPR)
as the associative recall engine to connect distant multi-hop evidence without iterative LLM calls.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations

from .models import Chunk
from .resolution import EntityResolver
from .text import tokens


def build_associative_graph(
    chunks: list[Chunk], resolver: EntityResolver | None = None
) -> tuple[dict[str, dict[str, float]], dict[str, set[str]], dict[str, list[str]]]:
    """Construct weighted entity co-occurrence adjacency graph and entity-chunk membership."""
    if resolver is None:
        resolver = EntityResolver.from_chunks(chunks)

    adjacency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    members: dict[str, set[str]] = defaultdict(set)
    chunk_entities: dict[str, list[str]] = {}

    for chunk in chunks:
        canon_entities = resolver.canonicalize(chunk.entities)
        chunk_entities[chunk.id] = canon_entities
        for entity in canon_entities:
            members[entity].add(chunk.id)
        for u, v in combinations(canon_entities, 2):
            adjacency[u][v] += 1.0
            adjacency[v][u] += 1.0

    return adjacency, members, chunk_entities


def compute_personalized_pagerank(
    adjacency: dict[str, dict[str, float]],
    personalization: dict[str, float],
    damping: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute stationary distribution of Random Walk with Restart (Personalized PageRank).

    Formula: r^(t+1) = damping * M * r^t + (1 - damping) * p
    """
    nodes = list(adjacency.keys())
    if not nodes or not personalization:
        return {}

    total_p = sum(personalization.values())
    if total_p <= 0:
        return {}
    p_vec = {u: personalization.get(u, 0.0) / total_p for u in nodes}

    # Precompute out-degree weights
    out_weights = {u: sum(neighbors.values()) for u, neighbors in adjacency.items()}

    # Initialize r^0 with personalization distribution
    r = dict(p_vec)

    for _ in range(max_iter):
        next_r = {u: (1.0 - damping) * p_vec[u] for u in nodes}
        for u in nodes:
            w_u = out_weights[u]
            if w_u > 0:
                share = damping * r[u] / w_u
                for v, weight in adjacency[u].items():
                    next_r[v] += share * weight
            else:
                # Dangling node teleports according to personalization
                for v in nodes:
                    next_r[v] += damping * r[u] * p_vec[v]

        diff = sum(abs(next_r[u] - r[u]) for u in nodes)
        r = next_r
        if diff < tol:
            break

    return r


def find_shortest_associative_path(
    seeds: list[str], target: str, adjacency: dict[str, dict[str, float]], max_depth: int = 3
) -> list[str]:
    """Find the shortest associative path from any seed to the target entity."""
    if target in seeds:
        return [target]

    queue = deque((s, [s]) for s in seeds if s in adjacency)
    visited = set(seeds)

    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        if len(path) <= max_depth:
            for nbr in sorted(adjacency.get(current, {}).keys()):
                if nbr not in visited:
                    visited.add(nbr)
                    if nbr == target:
                        return [*path, nbr]
                    queue.append((nbr, [*path, nbr]))

    return [seeds[0], target] if seeds else [target]


def hipporag_graph_scores(
    query: str,
    chunks: list[Chunk],
    damping: float = 0.85,
    resolver: EntityResolver | None = None,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Execute HippoRAG graph retrieval via Personalized PageRank associative activation."""
    if not chunks or not query.strip():
        return {}, {}

    if resolver is None:
        resolver = EntityResolver.from_chunks(chunks)

    adjacency, members, chunk_entities = build_associative_graph(chunks, resolver=resolver)
    if not adjacency:
        return {}, {}

    query_terms = set(tokens(query))
    seeds = resolver.resolve_query_seeds(query_terms, set(adjacency.keys()))
    if not seeds:
        seeds = sorted(entity for entity in adjacency if set(tokens(entity)) <= query_terms)[:12]
    if not seeds:
        return {}, {}

    # Personalization distribution over seeds
    personalization = {seed: 1.0 / len(seeds) for seed in seeds}
    ppr_ranks = compute_personalized_pagerank(adjacency, personalization, damping=damping)

    scores: dict[str, float] = {}
    paths: dict[str, list[str]] = {}

    for chunk in chunks:
        entities = chunk_entities.get(chunk.id, [])
        if not entities:
            continue

        # Score is sum of PPR activations for canonical entities in chunk
        chunk_activation = sum(ppr_ranks.get(e, 0.0) for e in entities)
        if chunk_activation > 0:
            scores[chunk.id] = chunk_activation
            # Best representative entity
            best_entity = max(entities, key=lambda e: ppr_ranks.get(e, 0.0))
            paths[chunk.id] = find_shortest_associative_path(seeds, best_entity, adjacency)

    if not scores:
        return {}, {}

    # Normalize scores to [0.0, 1.0]
    max_score = max(scores.values())
    normalized_scores = {cid: round(s / max_score, 4) for cid, s in scores.items()}

    return normalized_scores, paths
