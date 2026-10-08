"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).
Ref: Sarthi et al., "RAPTOR: Recursive Abstractive Processing for Tree-Organized
Retrieval" (Stanford University, ICLR 2024–2026).

Implements hierarchical semantic tree construction and collapsed tree retrieval:
1. Semantic Clustering: Dynamically clusters text chunks based on dense embedding
   cosine similarity or concept/token Jaccard overlap.
2. Abstractive/Extractive Recursive Summarization:
   - Layer 0: Original leaf chunks (granular details, exact citations, numeric facts).
   - Layer 1: Thematic cluster abstracts (intermediate semantic concepts grouping related chunks).
   - Layer 2: Root holistic summaries (broad cross-cutting document overview).
3. Collapsed Tree Indexing: Unifies multi-layer summaries and leaf chunks into a single
   multi-resolution retrieval candidate pool so queries can match both broad themes and
   specific factoids.
"""

from __future__ import annotations

import hashlib
import math
import re

from .models import Chunk
from .text import tokens

THEMATIC_PATTERNS = [
    r"\b(overview|summary|summarize|architecture|architectural|design|principles?|concept|concepts)\b",
    r"\b(high[- ]level|overall|holistic|system|workflow|lifecycle|structure|components?)\b",
    r"\b(compare|comparison|contrast|difference between|versus|\bvs\b)\b",
    r"\b(how does .* work|how do .* work|what is the relationship)\b",
    r"\b(main topics?|key themes?|comprehensive|all modules?|across)\b",
]


def is_raptor_thematic_query(query: str) -> bool:
    """Detect whether a query is thematic, architectural, comparative, or asking for broad synthesis."""
    q = query.strip().lower()
    return any(re.search(p, q) for p in THEMATIC_PATTERNS)


def compute_vector_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    return max(0.0, min(1.0, dot))


def compute_token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token Jaccard similarity between two texts."""
    set_a = set(tokens(text_a))
    set_b = set(tokens(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cluster_chunks(
    chunks: list[Chunk], cluster_size: int = 4, sim_threshold: float = 0.12
) -> list[list[Chunk]]:
    """Group chunks into coherent semantic clusters based on dense vectors or token overlap."""
    if not chunks:
        return []
    if len(chunks) <= cluster_size:
        return [list(chunks)]

    unassigned = list(chunks)
    clusters: list[list[Chunk]] = []

    has_vectors = all(len(c.vector) > 0 for c in chunks)

    while unassigned:
        seed = unassigned.pop(0)
        current_cluster = [seed]

        # Score remaining candidates against the seed
        scored_candidates: list[tuple[float, int, Chunk]] = []
        for idx, candidate in enumerate(unassigned):
            if has_vectors and len(candidate.vector) == len(seed.vector):
                sim = compute_vector_similarity(seed.vector, candidate.vector)
            else:
                sim = compute_token_jaccard(seed.text, candidate.text)

            # Document affinity bonus
            if candidate.document_id == seed.document_id:
                sim += 0.08

            if sim >= sim_threshold:
                scored_candidates.append((sim, idx, candidate))

        # Sort descending by similarity
        scored_candidates.sort(key=lambda item: -item[0])

        # Pick top candidates up to cluster_size - 1
        to_remove_indices = set()
        for _, idx, cand in scored_candidates[: cluster_size - 1]:
            current_cluster.append(cand)
            to_remove_indices.add(idx)

        # Remove assigned candidates in reverse order
        unassigned = [c for i, c in enumerate(unassigned) if i not in to_remove_indices]

        # If current_cluster is a single item and we have existing clusters, append to most similar cluster
        if len(current_cluster) == 1 and clusters:
            clusters[-1].append(seed)
        else:
            clusters.append(current_cluster)

    return clusters


def synthesize_cluster_summary(cluster: list[Chunk], layer: int) -> Chunk:
    """Synthesize an abstractive/extractive summary chunk for a cluster of child chunks."""
    first = cluster[0]

    # Extract distinct informative sentences from child chunks
    sentences: list[str] = []
    seen_hashes: set[str] = set()

    for chunk in cluster:
        parts = re.split(r"(?<=[.!?。！？])\s*|\n+", chunk.text)
        for part in parts:
            cleaned = part.strip()
            if len(cleaned) < 14 or cleaned.startswith("#"):
                continue
            h = hashlib.sha256(cleaned.lower().encode()).hexdigest()[:12]
            if h not in seen_hashes:
                seen_hashes.add(h)
                sentences.append(cleaned)
            if len(sentences) >= 6:
                break
        if len(sentences) >= 6:
            break

    if not sentences:
        sentences = [c.text[:200] for c in cluster[:3]]

    summary_body = " ".join(sentences)
    summary_text = f"[RAPTOR L{layer} Summary] {summary_body}"

    # Compute normalized centroid embedding vector across child chunks
    centroid_vector: list[float] = []
    if all(len(c.vector) > 0 for c in cluster):
        dim = len(first.vector)
        raw_centroid = [0.0] * dim
        for c in cluster:
            for i, val in enumerate(c.vector):
                raw_centroid[i] += val
        norm = math.sqrt(sum(v * v for v in raw_centroid))
        if norm > 0.0 and math.isfinite(norm):
            centroid_vector = [v / norm for v in raw_centroid]
        else:
            centroid_vector = [0.0] * dim

    # Combine unique entities
    merged_entities = sorted({e for c in cluster for e in c.entities})

    # Deterministic node ID
    cluster_digest = hashlib.sha256("".join(c.id for c in cluster).encode()).hexdigest()[:16]
    node_id = f"raptor:L{layer}:{cluster_digest}"

    return Chunk(
        id=node_id,
        document_id=first.document_id,
        filename=first.filename,
        text=summary_text,
        page=0,
        start=0,
        end=len(summary_text),
        vector=centroid_vector,
        entities=merged_entities,
    )


def build_raptor_layers(
    chunks: list[Chunk], max_layers: int = 2, cluster_size: int = 4
) -> dict[int, list[Chunk]]:
    """Build multi-level RAPTOR tree hierarchy (Layer 0 leaves, Layer 1+ abstracts)."""
    tree: dict[int, list[Chunk]] = {0: list(chunks)}
    if len(chunks) < 2 or max_layers < 1:
        return tree

    # Layer 1
    l1_clusters = cluster_chunks(chunks, cluster_size=cluster_size)
    l1_nodes = [synthesize_cluster_summary(c, layer=1) for c in l1_clusters if len(c) >= 2]
    tree[1] = l1_nodes

    # Layer 2 (Root holistic summary)
    if max_layers >= 2 and len(l1_nodes) >= 2:
        l2_clusters = cluster_chunks(l1_nodes, cluster_size=cluster_size)
        l2_nodes = [synthesize_cluster_summary(c, layer=2) for c in l2_clusters if len(c) >= 2]
        tree[2] = l2_nodes

    return tree


def expand_with_raptor_tree(
    chunks: list[Chunk], max_layers: int = 2, cluster_size: int = 4
) -> tuple[list[Chunk], dict[str, int]]:
    """Flatten RAPTOR hierarchy into unified collapsed tree candidate pool.

    Returns:
        (all_search_chunks, node_id_to_layer_map)
    """
    tree = build_raptor_layers(chunks, max_layers=max_layers, cluster_size=cluster_size)
    expanded = list(chunks)
    meta: dict[str, int] = {}

    for layer in range(1, max_layers + 1):
        for node in tree.get(layer, []):
            expanded.append(node)
            meta[node.id] = layer

    return expanded, meta
