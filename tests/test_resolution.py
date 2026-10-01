from __future__ import annotations

from rag_assistant.models import Chunk
from rag_assistant.resolution import (
    EntityResolver,
    calculate_entity_similarity,
    levenshtein_distance,
    normalize_entity_name,
)
from rag_assistant.retrieval import graph_scores


def test_normalize_entity_name():
    assert normalize_entity_name("  Postgres Databases  ") == "postgres database"
    assert normalize_entity_name("Redis Clusters") == "redis cluster"
    assert normalize_entity_name("Vector Indices") == "vector index"
    assert normalize_entity_name("Storage Nodes") == "storage node"


def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("postgres", "postgres") == 0
    assert levenshtein_distance("postgres", "postgre") == 1


def test_calculate_entity_similarity():
    assert calculate_entity_similarity("redis", "redis") == 1.0
    assert calculate_entity_similarity("postgresql", "postgres") >= 0.94
    assert calculate_entity_similarity("redis", "redis cluster") >= 0.90
    assert calculate_entity_similarity("failover", "fail-over") >= 0.85
    assert calculate_entity_similarity("apple", "banana") == 0.0


def test_entity_resolver_clustering():
    raw = ["PostgreSQL", "Postgres", "Redis", "Redis Cluster", "Atlas", "Beacon"]
    resolver = EntityResolver.from_entities(raw)

    # Postgres resolution
    canon_pg, conf_pg = resolver.resolve_with_confidence("postgresql")
    assert canon_pg == "postgres"
    assert conf_pg >= 0.90
    assert resolver.resolve("postgres") == "postgres"
    assert "postgresql" in resolver.get_aliases("postgres")

    # Redis resolution
    canon_redis, _ = resolver.resolve_with_confidence("redis cluster")
    assert canon_redis == "redis"
    assert "redis cluster" in resolver.get_aliases("redis")

    # Distinct entities remain intact
    assert resolver.resolve("atlas") == "atlas"
    assert resolver.resolve("beacon") == "beacon"

    # Canonicalize list
    canon_list = resolver.canonicalize(["PostgreSQL", "Postgres", "Redis Cluster"])
    assert canon_list == ["postgres", "redis"]


def test_entity_resolver_empty():
    resolver = EntityResolver.from_entities([])
    assert resolver.resolve("unknown") == "unknown"
    assert resolver.canonicalize([]) == []


def test_resolve_query_seeds():
    chunks = [
        Chunk(
            id="c1",
            document_id="d1",
            filename="a.txt",
            text="Atlas integrates with PostgreSQL for persistence.",
            page=1,
            start=0,
            end=50,
            entities=["Atlas", "PostgreSQL"],
        ),
        Chunk(
            id="c2",
            document_id="d2",
            filename="b.txt",
            text="Postgres provides transactional durability with Beacon.",
            page=1,
            start=0,
            end=55,
            entities=["Postgres", "Beacon"],
        ),
    ]
    resolver = EntityResolver.from_chunks(chunks)
    # Query with alias 'postgresql' matches canonical seed 'postgres'
    seeds = resolver.resolve_query_seeds({"postgresql"}, {"postgres", "atlas", "beacon"})
    assert "postgres" in seeds

    # Graph retrieval end-to-end: query using alias 'PostgreSQL' connects both c1 and c2
    scores, paths = graph_scores("How does PostgreSQL handle durability?", chunks, max_hops=2)
    assert "c1" in scores and "c2" in scores
    assert paths["c1"][0] == "postgres"
    assert paths["c2"][0] == "postgres"
