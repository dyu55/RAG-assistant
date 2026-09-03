"""
Entity Resolution & Canonicalization Engine for GraphRAG.
Provides entity normalization, acronym resolution, plural reduction,
and fuzzy alias disambiguation to merge synonymous entities into canonical graph nodes.
"""

from __future__ import annotations

import difflib
import logging
import re

logger = logging.getLogger(__name__)

# Standard domain acronyms and common synonym mappings
COMMON_ACRONYM_MAP: dict[str, str] = {
    "rag": "retrieval-augmented generation",
    "retrieval augmented generation": "retrieval-augmented generation",
    "llm": "large language model",
    "llms": "large language model",
    "large language models": "large language model",
    "kg": "knowledge graph",
    "kgs": "knowledge graph",
    "gnn": "graph neural network",
    "gnns": "graph neural network",
    "nlp": "natural language processing",
    "sota": "state of the art",
    "crag": "corrective rag",
    "corrective retrieval-augmented generation": "corrective rag",
    "cot": "chain of thought",
}

# Stopwords and noise determiners to strip from entity boundaries
NOISE_PREFIXES = ("the ", "a ", "an ", "this ", "these ", "those ")
NOISE_SUFFIXES = (" model", " architecture", " framework", " method", " approach")


class EntityResolver:
    """
    Resolves, normalizes, and disambiguates entity names for unified graph node merging.
    """

    def __init__(
        self,
        custom_aliases: dict[str, str] | None = None,
        enable_stemming: bool = True,
    ):
        self.aliases = dict(COMMON_ACRONYM_MAP)
        if custom_aliases:
            self.aliases.update(
                {k.lower().strip(): v.lower().strip() for k, v in custom_aliases.items()}
            )
        self.enable_stemming = enable_stemming

    def canonicalize(self, name: str) -> str:
        """
        Convert raw entity name to its canonical resolved form.

        Steps:
        1. Clean whitespace and non-alphanumeric boundaries
        2. Lowercase conversion
        3. Remove leading determiners (the, a, an)
        4. Apply acronym / domain alias resolution
        5. Normalize common plurals (e.g. transformers -> transformer)
        """
        if not name:
            return ""

        s = name.strip().lower()
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s)
        # Strip outer punctuation
        s = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", s, flags=re.UNICODE)

        # Strip leading noise prefixes
        for prefix in NOISE_PREFIXES:
            if s.startswith(prefix) and len(s) > len(prefix) + 2:
                s = s[len(prefix) :].strip()
                break

        # Check alias dictionary
        if s in self.aliases:
            return self.aliases[s]

        # Suffix / Plural normalization
        if self.enable_stemming:
            s = self._normalize_plural(s)

        # Check alias again after plural normalization
        if s in self.aliases:
            return self.aliases[s]

        return s

    def _normalize_plural(self, s: str) -> str:
        """Normalize common English plurals without aggressive over-stemming."""
        # words like 'embeddings' -> 'embedding', 'transformers' -> 'transformer'
        if s.endswith("ies") and len(s) > 5 and not s.endswith("series"):
            return s[:-3] + "y"
        if s.endswith("sses") and len(s) > 5:
            return s[:-2]
        if s.endswith("s") and len(s) > 3 and not s.endswith(("ss", "us", "is", "as", "os")):
            return s[:-1]
        return s

    def find_best_match(
        self,
        query_entity: str,
        candidate_entities: set[str] | list[str],
        threshold: float = 0.88,
    ) -> str | None:
        """
        Fuzzy match an entity name against a pool of existing candidate entities.
        Returns the closest matching canonical entity if similarity >= threshold, else None.
        """
        canonical = self.canonicalize(query_entity)
        if not canonical or not candidate_entities:
            return None

        if canonical in candidate_entities:
            return canonical

        # Compute fuzzy sequence matcher similarity
        matches = difflib.get_close_matches(
            canonical,
            list(candidate_entities),
            n=1,
            cutoff=threshold,
        )
        return matches[0] if matches else None


# Module singleton for quick utility access
default_resolver = EntityResolver()


def canonicalize_entity_name(name: str) -> str:
    """Convenience helper using the default EntityResolver singleton."""
    return default_resolver.canonicalize(name)
