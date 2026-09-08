from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from .models import Chunk, Evidence, Question
from .text import tokens


def keyword_scores(query: str, chunks: list[Chunk]) -> dict[str, float]:
    counts = [Counter(tokens(chunk.text)) for chunk in chunks]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / max(1, len(lengths)) or 1
    query_tokens = set(tokens(query))
    frequency = Counter(term for count in counts for term in query_tokens if term in count)
    result = {}
    for chunk, count, length in zip(chunks, counts, lengths, strict=True):
        score = 0.0
        for term in query_tokens:
            tf = count[term]
            if tf:
                idf = math.log(1 + (len(chunks) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average))
        if score > 0:
            result[chunk.id] = score
    return result


def vector_scores(vector: list[float], chunks: list[Chunk]) -> dict[str, float]:
    result = {}
    for chunk in chunks:
        if len(chunk.vector) != len(vector):
            raise ValueError(
                "Index embedding dimensions changed; reimport into a new data directory"
            )
        score = sum(a * b for a, b in zip(vector, chunk.vector, strict=True))
        if score > 0:
            result[chunk.id] = min(1.0, score)
    return result


def graph_index(chunks: list[Chunk]):
    members: dict[str, set[str]] = defaultdict(set)
    neighbors: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        for entity in chunk.entities:
            members[entity].add(chunk.id)
        for left, right in combinations(chunk.entities, 2):
            neighbors[left].add(right)
            neighbors[right].add(left)
    return members, neighbors


def graph_scores(query: str, chunks: list[Chunk], max_hops: int = 2):
    members, neighbors = graph_index(chunks)
    query_terms = set(tokens(query))
    seeds = sorted(entity for entity in members if set(tokens(entity)) <= query_terms)[:12]
    scores, paths = {}, {}
    queue = deque((seed, [seed]) for seed in seeds)
    visited = set(seeds)
    while queue:
        entity, path = queue.popleft()
        for chunk_id in sorted(members[entity]):
            score = 1 / len(path)
            if score > scores.get(chunk_id, 0):
                scores[chunk_id], paths[chunk_id] = score, path
        if len(path) <= max_hops:
            for neighbor in sorted(neighbors[entity]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))
                    if len(visited) >= 200:
                        break
    return scores, paths


def retrieve(question: Question, chunks: list[Chunk], vector: list[float] | None) -> list[Evidence]:
    if not chunks:
        return []
    with ThreadPoolExecutor(max_workers=3) as executor:
        keyword_job = executor.submit(keyword_scores, question.text, chunks)
        graph_job = executor.submit(graph_scores, question.text, chunks)
        vector_job = executor.submit(vector_scores, vector, chunks) if vector is not None else None
        keywords = keyword_job.result()
        graph, paths = graph_job.result()
        vectors = vector_job.result() if vector_job else {}
    available = {"keyword": keywords, "vector": vectors, "graph": graph}
    active = available if question.mode == "hybrid" else {question.mode: available[question.mode]}
    fusion: dict[str, float] = defaultdict(float)
    channels: dict[str, list[str]] = defaultdict(list)
    for channel, scores in active.items():
        for rank, (chunk_id, _) in enumerate(
            sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:60], 1
        ):
            fusion[chunk_id] += 1 / (60 + rank)
            channels[chunk_id].append(channel)
    lookup = {chunk.id: chunk for chunk in chunks}
    query_terms = set(tokens(question.text))
    selected = []
    for chunk_id, score in sorted(fusion.items(), key=lambda row: (-row[1], row[0])):
        chunk = lookup[chunk_id]
        coverage = len(query_terms & set(tokens(chunk.text))) / max(1, len(query_terms))
        relevance = min(1.0, max(coverage, vectors.get(chunk_id, 0)))
        if relevance < 0.12 and not graph.get(chunk_id):
            continue
        # Overlapping windows are useful for indexing, but should not crowd out other sources.
        if any(
            item.document_id == chunk.document_id
            and item.page == chunk.page
            and max(0, min(item.end, chunk.end) - max(item.start, chunk.start))
            > min(item.end - item.start, chunk.end - chunk.start) * 0.6
            for item in selected
        ):
            continue
        selected.append(
            Evidence(
                source_id=f"S{len(selected) + 1}",
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                text=chunk.text,
                page=chunk.page,
                start=chunk.start,
                end=chunk.end,
                relevance=round(relevance, 4),
                fusion_score=round(score, 6),
                channels=channels[chunk_id],
                path=paths.get(chunk_id, []),
            )
        )
        if len(selected) == question.top_k:
            break
    return selected


def graph_view(chunks: list[Chunk]) -> dict:
    members, _ = graph_index(chunks)
    top = sorted(members, key=lambda key: (-len(members[key]), key))[:45]
    edges = []
    for left, right in combinations(top, 2):
        shared = members[left] & members[right]
        if shared:
            edges.append(
                {
                    "source": left,
                    "target": right,
                    "weight": len(shared),
                    "chunk_ids": sorted(shared)[:5],
                    "relation": "co-occurs",
                }
            )
    return {
        "nodes": [{"id": entity, "count": len(members[entity])} for entity in top],
        "edges": sorted(edges, key=lambda edge: -edge["weight"])[:90],
        "description": "Entities that appear in the same passage. Connections are associations, not proven causal relationships.",
    }
