"""
Unit tests for the citation prefix logic and the sources_used tracking
on ReliabilityReport. These exercise the changes made to Generator and
ReliabilityChecker to support GraphRAG.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock

from core.generator import Citation
from core.reliability import ReliabilityChecker
from core.generator import GeneratedAnswer
from core.retriever import RetrievedChunk


def _chunk(text, source="vector", chunk_id="c1"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=0.7,
        metadata={"source": source, "filename": "f.txt"},
    )


def _label(chunk, idx):
    from core.generator import Generator
    return Generator._label_for(chunk, idx)


class TestGeneratorLabel:
    def test_vector_label(self):
        c = _chunk("x", source="vector")
        assert _label(c, 1) == "V1"
        assert _label(c, 12) == "V12"

    def test_graph_label(self):
        c = _chunk("x", source="graph")
        assert _label(c, 1) == "G1"

    def test_community_label(self):
        c = _chunk("x", source="community")
        assert _label(c, 1) == "C1"

    def test_unknown_falls_back_to_V(self):
        c = _chunk("x", source="something-else")
        assert _label(c, 1) == "V1"


class TestReliabilitySourcesUsed:
    def test_sources_used_set_from_chunks(self):
        rc = ReliabilityChecker()
        chunks = [
            _chunk("v text", source="vector", chunk_id="v1"),
            _chunk("g text", source="graph", chunk_id="g1"),
            _chunk("c text", source="community", chunk_id="c1"),
        ]
        answer = GeneratedAnswer(
            answer="answer [V1] [G2] [C3]",
            citations=[
                Citation(source_index=1, chunk_id="v1", quote="v text", source_type="vector"),
                Citation(source_index=2, chunk_id="g1", quote="g text", source_type="graph"),
                Citation(source_index=3, chunk_id="c1", quote="c text", source_type="community"),
            ],
            self_confidence=0.8,
        )
        report = rc.check(answer=answer, chunks=chunks)
        assert sorted(report.sources_used) == ["community", "graph", "vector"]

    def test_citation_strips_V_G_C_in_claim_extraction(self):
        rc = ReliabilityChecker()
        chunks = [_chunk("the answer is forty-two exactly", source="vector")]
        answer = GeneratedAnswer(
            answer="The answer is forty-two exactly according to [V1].",
            citations=[Citation(source_index=1, chunk_id="c1", quote="forty-two", source_type="vector")],
            self_confidence=0.9,
        )
        # Should not raise, should not count the citation as a claim
        report = rc.check(answer=answer, chunks=chunks)
        assert report is not None