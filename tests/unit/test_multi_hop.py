"""
Unit tests for graph/multi_hop.py (StepChain Multi-Hop Graph Reasoning Engine).
"""

from __future__ import annotations

from unittest.mock import Mock

from graph.multi_hop import (
    MultiHopPath,
    MultiHopReasoningEngine,
    ReasoningStep,
)


class TestReasoningStep:
    def test_step_formatting_without_description(self):
        step = ReasoningStep(source="GraphRAG", predicate="USES", target="Neo4j")
        assert step.to_readable() == "(GraphRAG) uses (Neo4j)"

    def test_step_formatting_with_description(self):
        step = ReasoningStep(
            source="GraphRAG",
            predicate="DEVELOPED_BY",
            target="Microsoft",
            description="Introduced in research paper 2024",
        )
        assert (
            "(GraphRAG) developed by (Microsoft) [Introduced in research paper 2024]"
            == step.to_readable()
        )


class TestMultiHopPath:
    def test_path_signature_and_hop_count(self):
        s1 = ReasoningStep(source="Transformer", predicate="INTRODUCES", target="Self-Attention")
        s2 = ReasoningStep(source="Self-Attention", predicate="ENABLES", target="BERT")
        path = MultiHopPath(steps=[s1, s2], total_weight=4.0)

        assert path.hop_count == 2
        assert (
            path.path_signature
            == "Transformer --[INTRODUCES]--> Self-Attention --[ENABLES]--> BERT"
        )

    def test_to_evidence_text_structure(self):
        s1 = ReasoningStep(source="A", predicate="LINKS_TO", target="B", description="A to B")
        path = MultiHopPath(steps=[s1], total_weight=1.0)
        evidence = path.to_evidence_text()

        assert "Multi-Hop Reasoning Chain (1 hops)" in evidence
        assert "Step 1: (A) links to (B) [A to B]" in evidence

    def test_empty_path(self):
        path = MultiHopPath()
        assert path.hop_count == 0
        assert path.path_signature == ""
        assert path.to_evidence_text() == ""


class TestMultiHopReasoningEngine:
    def test_build_path(self):
        engine = MultiHopReasoningEngine()
        raw_steps = [
            {
                "source": "X",
                "predicate": "CALLS",
                "target": "Y",
                "weight": 2.0,
                "description": "RPC call",
            },
            {
                "source": "Y",
                "predicate": "WRITES",
                "target": "Z",
                "weight": 3.0,
                "description": "DB commit",
            },
        ]
        path = engine.build_path(raw_steps)
        assert path.hop_count == 2
        assert path.total_weight == 5.0
        assert path.steps[0].source == "X"
        assert path.steps[1].target == "Z"

    def test_format_paths_as_chunks(self):
        engine = MultiHopReasoningEngine()
        s1 = ReasoningStep(source="A", predicate="RELATES", target="B", weight=2.0)
        p1 = MultiHopPath(steps=[s1], total_weight=2.0)

        chunks = engine.format_paths_as_chunks([p1], top_k=2)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "graph"
        assert chunks[0].metadata["is_multi_hop"] is True
        assert "Multi-Hop Reasoning Chain" in chunks[0].text
        assert 0.0 < chunks[0].score <= 1.0

    def test_traverse_subgraph_empty_without_neo4j(self):
        engine = MultiHopReasoningEngine(neo4j_client=None)
        assert engine.traverse_subgraph(["Entity1"]) == []

    def test_traverse_subgraph_with_mock_neo4j(self):
        mock_neo4j = Mock()
        mock_neo4j.execute_read.return_value = [
            {
                "step_dicts": [
                    {
                        "source": "LangChain",
                        "predicate": "INTEGRATES",
                        "target": "Neo4j",
                        "weight": 2.5,
                        "description": "Graph integration",
                    }
                ],
                "total_weight": 2.5,
            }
        ]
        engine = MultiHopReasoningEngine(neo4j_client=mock_neo4j)
        paths = engine.traverse_subgraph(["LangChain"])

        assert len(paths) == 1
        assert paths[0].steps[0].target == "Neo4j"
        assert mock_neo4j.execute_read.called
