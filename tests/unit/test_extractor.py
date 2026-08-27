"""
Unit tests for graph/extractor.py

Tests schema parsing, normalization, markdown-fence stripping, and
graceful failure handling. Uses a Mock provider so no API is called.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

from graph.extractor import (
    Entity,
    EntityRelationExtractor,
    ExtractionResult,
    _normalize,
    _strip_markdown_fences,
)

# ── _normalize ────────────────────────────────────────────────────────────────


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("GraphRAG") == "graphrag"

    def test_collapses_whitespace(self):
        assert _normalize("foo   bar") == "foo bar"

    def test_strips_punctuation(self):
        assert _normalize("...hello?") == "hello"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_none_safe(self):
        assert _normalize(None) == ""

    def test_unicode(self):
        # Non-ASCII letters should be preserved (no aggressive transliteration)
        result = _normalize("café résumé")
        assert "café" in result
        assert "résumé" in result


# ── _strip_markdown_fences ────────────────────────────────────────────────────


class TestStripFences:
    def test_json_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(raw) == '{"a": 1}'

    def test_plain_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert _strip_markdown_fences(raw) == '{"a": 1}'

    def test_no_fence(self):
        assert _strip_markdown_fences('{"a":1}') == '{"a":1}'

    def test_empty(self):
        assert _strip_markdown_fences("") == ""


# ── EntityRelationExtractor ──────────────────────────────────────────────────


def _mock_provider_returning(payload: dict):
    provider = Mock()
    provider.generate_json.return_value = payload
    return provider


def _golden_payload():
    return {
        "entities": [
            {
                "name": "GraphRAG",
                "type": "TECHNOLOGY",
                "description": "Microsoft's graph-based RAG.",
            },
            {"name": "Microsoft", "type": "ORG", "description": "Technology company."},
            {"name": "", "type": "CONCEPT", "description": "should be skipped"},
        ],
        "relations": [
            {
                "source": "GraphRAG",
                "target": "Microsoft",
                "predicate": "introduced_by",
                "description": "GraphRAG was introduced by Microsoft.",
            },
            {"source": "", "target": "X", "predicate": "USES", "description": "missing source"},
            {"source": "X", "target": "", "predicate": "USES", "description": "missing target"},
            {"source": "X", "target": "Y", "predicate": "", "description": "missing predicate"},
        ],
    }


class TestExtractorParse:
    def test_happy_path(self):
        provider = _mock_provider_returning(_golden_payload())
        extractor = EntityRelationExtractor(provider)
        result = extractor.extract_chunk("GraphRAG was introduced by Microsoft in 2024.")

        assert isinstance(result, ExtractionResult)
        names = sorted(e.name for e in result.entities)
        assert names == ["GraphRAG", "Microsoft"]
        assert all(isinstance(e, Entity) for e in result.entities)
        # Self-loop relation should be dropped (source==target normalization).
        assert len(result.relations) == 1
        rel = result.relations[0]
        assert rel.predicate == "INTRODUCED_BY"  # normalized to upper

    def test_string_payload_with_fences(self):
        provider = Mock()
        provider.generate_json.side_effect = lambda **kwargs: json.loads(
            _strip_markdown_fences("```json\n" + json.dumps(_golden_payload()) + "\n```")
        )
        extractor = EntityRelationExtractor(provider)
        result = extractor.extract_chunk("some text")
        assert len(result.entities) == 2

    def test_dedupes_entities(self):
        payload = {
            "entities": [
                {"name": "GraphRAG", "type": "TECHNOLOGY"},
                {"name": "graphrag", "type": "TECHNOLOGY"},
                {"name": "  GRAPHRAG  ", "type": "TECHNOLOGY"},
            ],
            "relations": [],
        }
        extractor = EntityRelationExtractor(_mock_provider_returning(payload))
        result = extractor.extract_chunk("x")
        assert len(result.entities) == 1
        assert result.entities[0].normalized_name == "graphrag"

    def test_empty_text_returns_empty(self):
        extractor = EntityRelationExtractor(Mock())
        assert extractor.extract_chunk("") == ExtractionResult()
        assert extractor.extract_chunk("   ") == ExtractionResult()

    def test_provider_failure_returns_empty(self):
        provider = Mock()
        provider.generate_json.side_effect = Exception("boom")
        extractor = EntityRelationExtractor(provider)
        result = extractor.extract_chunk("anything")
        assert result == ExtractionResult()

    def test_malformed_payload_raises(self):
        # Not a dict and not even JSON-parseable
        provider = Mock()
        provider.generate_json.side_effect = ValueError("not a dict")
        extractor = EntityRelationExtractor(provider)
        # After retries, returns empty result rather than propagating
        assert extractor.extract_chunk("text") == ExtractionResult()

    def test_missing_fields_skipped(self):
        payload = {
            "entities": [
                {"type": "TECHNOLOGY", "description": "no name"},  # no name
                {"name": "OK", "type": "X"},
            ],
            "relations": [
                {"predicate": "USES"},  # no source/target
                {"source": "OK", "target": "OK2", "predicate": "USES"},
                {"source": "", "target": "OK2", "predicate": "USES"},  # empty src
            ],
        }
        extractor = EntityRelationExtractor(_mock_provider_returning(payload))
        result = extractor.extract_chunk("x")
        assert len(result.entities) == 1
        assert result.entities[0].name == "OK"
        assert len(result.relations) == 1

    def test_extract_query_entities(self):
        extractor = EntityRelationExtractor(_mock_provider_returning(_golden_payload()))
        ents = extractor.extract_query_entities("what does graphrag say?")
        assert any(e.name == "GraphRAG" for e in ents)
