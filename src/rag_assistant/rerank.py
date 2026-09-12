from __future__ import annotations

import re

from .models import Chunk
from .text import entities, tokens


def analyze_query_weights(query: str) -> dict[str, float]:
    """Dynamically balance retrieval channel weights (keyword, vector, graph) based on query characteristics."""
    # Base weights
    w_keyword = 1.0
    w_vector = 1.0
    w_graph = 1.0

    # Check for explicit quotes
    has_quotes = bool(re.search(r'"[^"]+"', query))
    if has_quotes:
        w_keyword += 0.8

    # Check for technical identifiers (camelCase, snake_case, code tokens, versions)
    has_identifiers = bool(
        re.search(r"\b[a-z0-9]+_[a-z0-9_]+\b|\b[A-Z]+_[A-Z0-9_]+\b|\bv?\d+\.\d+\b", query)
    )
    if has_identifiers:
        w_keyword += 0.6

    # Check for entity / relational intent (e.g. who, connects, relates, between, graph)
    query_entities = entities(query)
    has_relational = bool(
        re.search(
            r"\b(connects?|relationships?|between|associated|linked|depends?|who|parent|child)\b",
            query,
            re.IGNORECASE,
        )
    )
    if query_entities or has_relational:
        w_graph += 0.8 if has_relational else 0.5

    # Check for conceptual / long explanatory query intent
    words = query.split()
    is_conceptual = bool(
        re.search(r"^(why|how|explain|describe|what is|summarize)\b", query, re.IGNORECASE)
    )
    if is_conceptual or len(words) >= 8:
        w_vector += 0.6

    # Normalize weights so sum is 3.0
    total = w_keyword + w_vector + w_graph
    scale = 3.0 / total
    return {
        "keyword": round(w_keyword * scale, 3),
        "vector": round(w_vector * scale, 3),
        "graph": round(w_graph * scale, 3),
    }


def score_phrase_match(query: str, text: str) -> float:
    """Calculate exact substring and n-gram phrase alignment between query and candidate passage."""
    q_norm = " ".join(re.findall(r"[a-z0-9]+", query.casefold()))
    t_norm = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

    if not q_norm or not t_norm:
        return 0.0

    # Direct substring match
    if q_norm in t_norm:
        return 1.0

    # Bigram and trigram phrase overlap
    q_words = q_norm.split()
    if len(q_words) < 2:
        return 1.0 if q_words[0] in t_norm else 0.0

    bigrams = [f"{q_words[i]} {q_words[i + 1]}" for i in range(len(q_words) - 1)]
    bigram_matches = sum(1 for bg in bigrams if bg in t_norm)
    score = bigram_matches / len(bigrams)

    # If trigrams exist, boost if matched
    if len(q_words) >= 3:
        trigrams = [
            f"{q_words[i]} {q_words[i + 1]} {q_words[i + 2]}" for i in range(len(q_words) - 2)
        ]
        trigram_matches = sum(1 for tg in trigrams if tg in t_norm)
        score = 0.6 * score + 0.4 * (trigram_matches / len(trigrams))

    return min(1.0, round(score, 4))


def score_term_proximity(query_terms: set[str], text: str) -> float:
    """Measure the compactness (window span in words) of query terms inside the candidate text."""
    text_words = [w for w in re.findall(r"[a-z0-9_]+", text.casefold())]
    if not text_words or not query_terms:
        return 0.0

    matched_positions = [i for i, w in enumerate(text_words) if w in query_terms]
    if len(matched_positions) <= 1:
        return 0.5 if matched_positions else 0.0

    # Find the tightest window span containing matched query terms
    min_span = len(text_words)
    target_count = min(len(query_terms), 3)

    for i in range(len(matched_positions) - target_count + 1):
        span = matched_positions[i + target_count - 1] - matched_positions[i] + 1
        if span < min_span:
            min_span = span

    # Compact span (< 15 words) gets ~1.0; wide span (> 60 words) degrades gracefully
    proximity = max(0.0, 1.0 - (min_span - target_count) / max(1, 50))
    return min(1.0, round(proximity, 4))


def score_entity_alignment(query: str, chunk_entities: list[str], path: list[str]) -> float:
    """Evaluate entity matching and graph path continuity between query and chunk."""
    q_entities = set(entities(query))
    if not q_entities:
        return 0.5  # Neutral when query doesn't specify entities

    c_entities = set(normalize_entity(e) for e in chunk_entities)
    overlap = len(q_entities & c_entities)
    if overlap > 0:
        return min(1.0, 0.7 + 0.3 * (overlap / len(q_entities)))

    p_entities = set(normalize_entity(e) for e in path)
    if len(q_entities & p_entities) > 0:
        return 0.75

    return 0.2


def normalize_entity(e: str) -> str:
    return " ".join(e.casefold().split())


def compute_rerank_score(
    query: str,
    chunk: Chunk,
    vector_score: float = 0.0,
    graph_score: float = 0.0,
    path: list[str] | None = None,
) -> float:
    """Cross-feature semantic reranking score combining lexical coverage, phrase matching,

    term proximity, entity alignment, and dense vector similarity.
    """
    q_tokens = set(tokens(query))
    c_tokens = set(tokens(chunk.text))
    lexical_coverage = len(q_tokens & c_tokens) / max(1, len(q_tokens))

    if lexical_coverage == 0.0 and vector_score <= 0.0 and graph_score <= 0.0:
        return 0.0

    phrase_score = score_phrase_match(query, chunk.text)
    proximity_score = score_term_proximity(q_tokens & c_tokens, chunk.text)
    entity_score = score_entity_alignment(query, chunk.entities, path or [])

    v_score = min(1.0, max(0.0, vector_score))
    if v_score > 0.0:
        score = (
            0.25 * lexical_coverage
            + 0.25 * v_score
            + 0.20 * phrase_score
            + 0.15 * proximity_score
            + 0.15 * entity_score
        )
    else:
        score = (
            0.35 * lexical_coverage
            + 0.25 * phrase_score
            + 0.20 * proximity_score
            + 0.20 * entity_score
        )

    # Graph hop bonus if reachable
    if graph_score > 0.0:
        score += min(0.1, 0.05 * graph_score)

    return round(min(1.0, max(0.0, score)), 4)
