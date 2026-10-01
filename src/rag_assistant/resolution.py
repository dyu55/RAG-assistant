"""
Graph Entity Resolution & Semantic Canonicalization Engine.
Implements 2026 SOTA GraphRAG entity resolution standards:
1. Probabilistic and rule-based entity matching (stemming, alias dictionary, containment, edit distance).
2. Disjoint Set Union (DSU) canonical clustering to create a single source of truth for each entity.
3. Preserves alias confidence scores on graph mappings to eliminate graph fragmentation.
4. Seamlessly expands query seed matching across entity aliases.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .models import Chunk
from .text import tokens

KNOWN_ALIASES: dict[str, str] = {
    "postgresql": "postgres",
    "postgres db": "postgres",
    "postgres database": "postgres",
    "redis cache": "redis",
    "redis cluster": "redis",
    "k8s": "kubernetes",
    "es": "elasticsearch",
    "llm": "large language model",
    "ann": "approximate nearest neighbor",
    "kg": "knowledge graph",
}

SUFFIX_REPLACEMENTS = (
    (r"databases$", "database"),
    (r"clusters$", "cluster"),
    (r"servers$", "server"),
    (r"gateways$", "gateway"),
    (r"indices$", "index"),
    (r"indexes$", "index"),
    (r"shards$", "shard"),
    (r"services$", "service"),
    (r"caches$", "cache"),
    (r"nodes$", "node"),
    (r"queries$", "query"),
    (r"connections$", "connection"),
)


def normalize_entity_name(name: str) -> str:
    """Normalize raw entity string by stripping whitespace, case-folding, and stemming plurals."""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    for pat, repl in SUFFIX_REPLACEMENTS:
        if re.search(pat, cleaned):
            cleaned = re.sub(pat, repl, cleaned)
            break
    return cleaned


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_entity_similarity(e1: str, e2: str) -> float:
    """Calculate match probability between two normalized entity candidates (0.0 to 1.0)."""
    if e1 == e2:
        return 1.0

    # Known alias check
    if KNOWN_ALIASES.get(e1) == e2 or KNOWN_ALIASES.get(e2) == e1:
        return 0.98

    # Prefix match for substantial words
    if (e1.startswith(e2) or e2.startswith(e1)) and min(len(e1), len(e2)) >= 5:
        return 0.94

    t1 = set(tokens(e1))
    t2 = set(tokens(e2))
    if t1 and t2:
        # Full token subset containment (e.g. "redis" in "redis cluster")
        if t1 < t2 or t2 < t1:
            return 0.90
        # Token Jaccard
        jaccard = len(t1 & t2) / len(t1 | t2)
        if jaccard >= 0.70:
            return round(jaccard, 3)

    # Edit distance for small typos/variations (length >= 5)
    min_len = min(len(e1), len(e2))
    if min_len >= 5:
        dist = levenshtein_distance(e1, e2)
        if dist == 1:
            return 0.88
        if dist == 2 and min_len >= 8:
            return 0.80

    return 0.0


class EntityResolver:
    """Disjoint-Set Union (DSU) based entity resolution and canonical identity registry."""

    def __init__(
        self,
        mappings: dict[str, str],
        confidences: dict[str, float],
        aliases: dict[str, set[str]],
    ):
        self._mappings = mappings
        self._confidences = confidences
        self._aliases = aliases

    @classmethod
    def from_entities(
        cls, raw_entities: list[str] | set[str], threshold: float = 0.85
    ) -> EntityResolver:
        cleaned_list = sorted(
            {normalize_entity_name(e) for e in raw_entities if len(e.strip()) > 1}
        )
        if not cleaned_list:
            return cls({}, {}, {})

        # Disjoint Set Union (DSU)
        parent = {e: e for e in cleaned_list}

        def find(item: str) -> str:
            if parent[item] != item:
                parent[item] = find(parent[item])
            return parent[item]

        def union(i1: str, i2: str):
            r1, r2 = find(i1), find(i2)
            if r1 != r2:
                # Prioritize shorter, cleaner root name as cluster parent
                if len(r1) <= len(r2):
                    parent[r2] = r1
                else:
                    parent[r1] = r2

        pair_confidences: dict[tuple[str, str], float] = {}
        for i in range(len(cleaned_list)):
            for j in range(i + 1, len(cleaned_list)):
                e1, e2 = cleaned_list[i], cleaned_list[j]
                sim = calculate_entity_similarity(e1, e2)
                if sim >= threshold:
                    union(e1, e2)
                    pair_confidences[(e1, e2)] = sim
                    pair_confidences[(e2, e1)] = sim

        # Group by root
        groups: dict[str, list[str]] = defaultdict(list)
        for e in cleaned_list:
            groups[find(e)].append(e)

        mappings: dict[str, str] = {}
        confidences: dict[str, float] = {}
        aliases: dict[str, set[str]] = defaultdict(set)

        for _, members in groups.items():
            # Pick canonical name: preference for single word or shortest name
            canonical = sorted(members, key=lambda m: (len(m.split()), len(m), m))[0]
            for member in members:
                mappings[member] = canonical
                conf = (
                    1.0 if member == canonical else pair_confidences.get((member, canonical), 0.90)
                )
                confidences[member] = conf
                aliases[canonical].add(member)

        return cls(mappings=mappings, confidences=confidences, aliases=aliases)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], threshold: float = 0.85) -> EntityResolver:
        all_entities = [e for c in chunks for e in c.entities]
        return cls.from_entities(all_entities, threshold=threshold)

    def resolve(self, entity: str) -> str:
        """Resolve an entity string to its canonical representation."""
        norm = normalize_entity_name(entity)
        return self._mappings.get(norm, norm)

    def resolve_with_confidence(self, entity: str) -> tuple[str, float]:
        norm = normalize_entity_name(entity)
        canonical = self._mappings.get(norm, norm)
        conf = self._confidences.get(norm, 1.0 if canonical == norm else 0.85)
        return canonical, conf

    def get_aliases(self, canonical: str) -> set[str]:
        """Return all known aliases for a canonical entity."""
        norm = normalize_entity_name(canonical)
        return self._aliases.get(norm, {norm})

    def canonicalize(self, entities: list[str]) -> list[str]:
        """Convert a list of raw entities into unique canonical names."""
        result = set()
        for e in entities:
            if len(e.strip()) > 1:
                result.add(self.resolve(e))
        return sorted(result)

    def resolve_query_seeds(self, query_terms: set[str], members: dict[str, set[str]]) -> list[str]:
        """Find matching seed canonical entities using both canonical forms and their aliases."""
        matched_seeds = set()
        for canonical in members:
            # 1. Direct match with canonical tokens
            c_tokens = set(tokens(canonical))
            if c_tokens and c_tokens <= query_terms:
                matched_seeds.add(canonical)
                continue

            # 2. Match through any alias tokens
            for alias in self.get_aliases(canonical):
                a_tokens = set(tokens(alias))
                if a_tokens and a_tokens <= query_terms:
                    matched_seeds.add(canonical)
                    break

        return sorted(matched_seeds)[:12]
