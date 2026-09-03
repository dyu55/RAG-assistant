"""
Unit tests for graph/entity_resolver.py (Entity Resolution & Canonicalization Engine).
"""

from __future__ import annotations

from graph.entity_resolver import EntityResolver, canonicalize_entity_name


class TestEntityResolver:
    def test_canonicalize_empty_and_whitespace(self):
        resolver = EntityResolver()
        assert resolver.canonicalize("") == ""
        assert resolver.canonicalize("   \n\t  ") == ""

    def test_canonicalize_acronyms(self):
        resolver = EntityResolver()
        assert resolver.canonicalize("RAG") == "retrieval-augmented generation"
        assert (
            resolver.canonicalize("Retrieval Augmented Generation")
            == "retrieval-augmented generation"
        )
        assert resolver.canonicalize("LLM") == "large language model"
        assert resolver.canonicalize("LLMs") == "large language model"
        assert resolver.canonicalize("KG") == "knowledge graph"
        assert resolver.canonicalize("CRAG") == "corrective rag"
        assert resolver.canonicalize("GraphRAG") == "graphrag"

    def test_canonicalize_plural_reduction(self):
        resolver = EntityResolver()
        assert resolver.canonicalize("transformers") == "transformer"
        assert resolver.canonicalize("embeddings") == "embedding"
        assert resolver.canonicalize("agents") == "agent"
        assert resolver.canonicalize("databases") == "database"

    def test_canonicalize_strips_determiner_prefixes(self):
        resolver = EntityResolver()
        assert resolver.canonicalize("The Transformer") == "transformer"
        assert resolver.canonicalize("an attention mechanism") == "attention mechanism"
        assert resolver.canonicalize("a neural network") == "neural network"

    def test_custom_aliases_override(self):
        resolver = EntityResolver(custom_aliases={"ag": "antigravity"})
        assert resolver.canonicalize("AG") == "antigravity"

    def test_find_best_match_exact_and_fuzzy(self):
        resolver = EntityResolver()
        candidates = {"transformer", "convolutional neural network", "recurrent neural network"}

        # Exact canonical match
        assert resolver.find_best_match("the transformers", candidates) == "transformer"

        # Fuzzy match
        assert resolver.find_best_match("transformerr", candidates, threshold=0.8) == "transformer"

        # No match when too dissimilar
        assert resolver.find_best_match("random unrelated concept", candidates) is None

    def test_canonicalize_entity_name_helper(self):
        assert canonicalize_entity_name("GraphRAG") == "graphrag"
        assert canonicalize_entity_name("The LLMs") == "large language model"
