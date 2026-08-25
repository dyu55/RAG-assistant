"""
Unit tests for graph/router.py

Tests explicit-mode override, heuristic fallback, and LLM parsing.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock

from graph.router import QueryRouter, RouteDecision, RouteMode


class TestExplicitMode:
    def test_pinned_local(self):
        r = QueryRouter(provider=Mock(), mode_setting="local")
        d = r.route("anything")
        assert d.mode == RouteMode.LOCAL
        assert d.from_setting is True

    def test_pinned_global(self):
        r = QueryRouter(provider=Mock(), mode_setting="global")
        d = r.route("anything")
        assert d.mode == RouteMode.GLOBAL
        assert d.run_graph_global is True
        assert d.run_vector is False

    def test_pinned_both(self):
        r = QueryRouter(provider=Mock(), mode_setting="both")
        d = r.route("anything")
        assert d.run_vector is True
        assert d.run_graph_local is True
        assert d.run_graph_global is True

    def test_pinned_off(self):
        r = QueryRouter(provider=Mock(), mode_setting="off")
        d = r.route("anything")
        assert d.mode == RouteMode.OFF
        assert d.run_vector is True
        assert d.run_graph_local is False
        assert d.run_graph_global is False

    def test_auto_triggers_llm(self):
        provider = Mock()
        provider.generate_json.return_value = {
            "mode": "GLOBAL",
            "confidence": 0.9,
            "reason": "summarize-y",
        }
        r = QueryRouter(provider=provider, mode_setting="auto")
        d = r.route("summarize the corpus")
        assert d.mode == RouteMode.GLOBAL
        assert d.from_setting is False
        assert d.confidence == 0.9


class TestHeuristicFallback:
    def test_no_provider_uses_heuristic(self):
        r = QueryRouter(provider=None, mode_setting="auto")
        d = r.route("summarize the main themes")
        assert d.mode == RouteMode.GLOBAL
        assert "Heuristic" in d.reason

    def test_local_default_for_specific_question(self):
        r = QueryRouter(provider=None, mode_setting="auto")
        d = r.route("what does the spec say about rate limits?")
        assert d.mode == RouteMode.LOCAL

    def test_global_when_keyword_matches(self):
        r = QueryRouter(provider=None, mode_setting="auto")
        for q in [
            "summarize the documents",
            "give me a high-level overview",
            "what are the main themes?",
            "what patterns emerge from the corpus?",
        ]:
            d = r.route(q)
            assert d.mode == RouteMode.GLOBAL, q


class TestParse:
    def test_invalid_mode_falls_back_to_local(self):
        r = QueryRouter(provider=Mock(), mode_setting="auto")
        d = r._parse({"mode": "garbage"})
        assert d.mode == RouteMode.LOCAL

    def test_confidence_clamped(self):
        r = QueryRouter(provider=Mock(), mode_setting="auto")
        d = r._parse({"mode": "LOCAL", "confidence": 5.0})
        assert d.confidence == 1.0
        d = r._parse({"mode": "LOCAL", "confidence": -2.0})
        assert d.confidence == 0.0

    def test_string_payload(self):
        r = QueryRouter(provider=Mock(), mode_setting="auto")
        d = r._parse('{"mode": "BOTH", "confidence": 0.6, "reason": "compare"}')
        assert d.mode == RouteMode.BOTH
        assert d.run_vector is True
        assert d.run_graph_global is True


class TestRouteDecisionSerialization:
    def test_to_dict_contains_bools(self):
        d = RouteDecision(mode=RouteMode.LOCAL, confidence=0.8, reason="r")
        out = d.to_dict()
        assert out["mode"] == "local"
        assert out["run_vector"] is True
        assert out["run_graph_local"] is True
        assert out["run_graph_global"] is False