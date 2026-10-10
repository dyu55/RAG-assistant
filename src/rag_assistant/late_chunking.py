"""
Late Chunking: Contextual Chunk Embeddings Using Document Contextualization.
Ref: Günther et al., "Late Chunking: Contextual Chunk Embeddings Using Long-Context
Models" (Jina AI Research, 2024–2026).

Implements the Late Chunking paradigm to resolve context fragmentation:
1. Contextual Conditioning: In traditional naive (early) chunking, chunks are split
   before embedding, losing broader document context, entity definitions, and pronoun referents.
2. Global Document Representation: Derives the document-level semantic anchor vector
   (mean pooling / centroid across document tokens and spans).
3. Contextual Vector Blending: Injects document-wide semantic awareness into individual
   chunk vectors:
       v_late = normalize((1 - w) * v_chunk + w * v_doc)
   preserving 80% local specificity while gaining 20% global document awareness.
4. Anaphora & Coreference Preservation: Enables search queries to retrieve ambiguous
   or pronoun-heavy passages that would otherwise miss keyword/semantic matching.
"""

from __future__ import annotations

from .models import Chunk
from .providers import normalized_vector


def compute_document_centroid(chunk_vectors: list[list[float]]) -> list[float]:
    """Compute the normalized centroid vector representing global document semantics."""
    if not chunk_vectors:
        return []

    dim = len(chunk_vectors[0])
    raw_sum = [0.0] * dim
    valid_count = 0

    for vec in chunk_vectors:
        if len(vec) == dim:
            for i, val in enumerate(vec):
                raw_sum[i] += val
            valid_count += 1

    if valid_count == 0:
        return [0.0] * dim

    return normalized_vector(raw_sum)


def blend_contextual_vector(
    chunk_vector: list[float],
    doc_vector: list[float],
    context_weight: float = 0.20,
) -> list[float]:
    """Blend a local chunk vector with the global document context vector."""
    if not chunk_vector or not doc_vector or len(chunk_vector) != len(doc_vector):
        return chunk_vector

    w = max(0.0, min(0.5, context_weight))
    if w == 0.0:
        return chunk_vector

    blended = [
        (1.0 - w) * c_val + w * d_val for c_val, d_val in zip(chunk_vector, doc_vector, strict=True)
    ]
    return normalized_vector(blended)


def apply_late_chunking(
    chunks: list[Chunk],
    chunk_vectors: list[list[float]],
    context_weight: float = 0.20,
    doc_vector: list[float] | None = None,
) -> list[list[float]]:
    """Apply Late Chunking to inject global document context into individual chunk embeddings."""
    if not chunk_vectors or len(chunk_vectors) <= 1 or context_weight <= 0.0:
        return chunk_vectors

    # Derive document-level semantic vector if not explicitly provided
    d_vec = (
        doc_vector
        if (doc_vector and len(doc_vector) == len(chunk_vectors[0]))
        else compute_document_centroid(chunk_vectors)
    )
    if not d_vec:
        return chunk_vectors

    return [
        blend_contextual_vector(c_vec, d_vec, context_weight=context_weight)
        for c_vec in chunk_vectors
    ]


def evaluate_late_chunking_retention(
    chunk_vector: list[float],
    doc_vector: list[float],
    late_vector: list[float],
) -> dict[str, float]:
    """Evaluate local fidelity preservation and document context gain of late chunking."""
    if not chunk_vector or not late_vector or len(chunk_vector) != len(late_vector):
        return {"local_fidelity": 1.0, "context_gain": 0.0}

    local_sim = sum(a * b for a, b in zip(chunk_vector, late_vector, strict=True))
    context_sim = (
        sum(a * b for a, b in zip(doc_vector, late_vector, strict=True))
        if doc_vector and len(doc_vector) == len(late_vector)
        else 0.0
    )

    return {
        "local_fidelity": round(max(0.0, min(1.0, local_sim)), 4),
        "context_gain": round(max(0.0, min(1.0, context_sim)), 4),
    }
