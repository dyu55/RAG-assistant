"""
Multi-Hop Graph Reasoning & Subgraph Traversal Engine (StepChain GraphRAG).
Discovers multi-hop associative and causal paths across knowledge graph entities,
synthesizing multi-step reasoning evidence chains to answer complex, interconnected queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single hop transition in a multi-hop evidence path."""

    source: str
    predicate: str
    target: str
    description: str = ""
    weight: float = 1.0

    def to_readable(self) -> str:
        pred = self.predicate.replace("_", " ").lower()
        if self.description:
            return f"({self.source}) {pred} ({self.target}) [{self.description}]"
        return f"({self.source}) {pred} ({self.target})"


@dataclass
class MultiHopPath:
    """A multi-hop causal/associative reasoning path through the graph."""

    steps: list[ReasoningStep] = field(default_factory=list)
    total_weight: float = 0.0

    @property
    def hop_count(self) -> int:
        return len(self.steps)

    @property
    def path_signature(self) -> str:
        if not self.steps:
            return ""
        nodes = [self.steps[0].source]
        for s in self.steps:
            nodes.append(f"--[{s.predicate}]-->")
            nodes.append(s.target)
        return " ".join(nodes)

    def to_evidence_text(self) -> str:
        """Render the reasoning chain as explicit natural language context."""
        if not self.steps:
            return ""
        lines = [f"Multi-Hop Reasoning Chain ({self.hop_count} hops): {self.path_signature}"]
        for idx, step in enumerate(self.steps, 1):
            detail = f" - Step {idx}: {step.to_readable()}"
            lines.append(detail)
        return "\n".join(lines)


class MultiHopReasoningEngine:
    """
    Traverses multi-hop entity relations in the knowledge graph to build
    structured evidence chains for multi-step reasoning.
    """

    def __init__(self, neo4j_client=None):
        self.neo4j = neo4j_client

    def build_path(self, raw_steps: list[dict]) -> MultiHopPath:
        """Construct a MultiHopPath from a sequence of raw step dictionaries."""
        steps = []
        total_w = 0.0
        for s in raw_steps:
            w = float(s.get("weight", 1.0))
            steps.append(
                ReasoningStep(
                    source=s.get("source", ""),
                    predicate=s.get("predicate", "RELATED"),
                    target=s.get("target", ""),
                    description=s.get("description", ""),
                    weight=w,
                )
            )
            total_w += w
        return MultiHopPath(steps=steps, total_weight=total_w)

    def format_paths_as_chunks(
        self,
        paths: list[MultiHopPath],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Convert multi-hop paths into RetrievedChunks compatible with the pipeline."""
        chunks: list[RetrievedChunk] = []
        # Sort paths by hop count ascending (parsimonious) and weight descending
        sorted_paths = sorted(paths, key=lambda p: (-p.total_weight, p.hop_count))[:top_k]

        for idx, path in enumerate(sorted_paths, 1):
            if not path.steps:
                continue
            text = path.to_evidence_text()
            score = min(1.0, max(0.1, path.total_weight / (path.hop_count * 2.0)))
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"multihop:{path.steps[0].source}:{path.steps[-1].target}:{idx}",
                    text=text,
                    score=round(score, 4),
                    metadata={
                        "source": "graph",
                        "is_multi_hop": True,
                        "hop_count": path.hop_count,
                        "path_signature": path.path_signature,
                    },
                )
            )
        return chunks

    def traverse_subgraph(
        self,
        focal_entities: list[str],
        max_hops: int = 2,
        limit: int = 10,
    ) -> list[MultiHopPath]:
        """
        Execute multi-hop traversal in Neo4j starting from focal entities.
        Falls back to empty list if neo4j is not connected or query yields no paths.
        """
        if not self.neo4j or not focal_entities:
            return []

        try:
            # Cypher multi-hop pattern matching (1..2 hops)
            rows = self.neo4j.execute_read(
                """
                UNWIND $focals AS start_name
                MATCH path = (start:Entity {name: start_name})-[:RELATED*1..2]-(dest:Entity)
                WHERE start <> dest
                WITH path,
                     [rel IN relationships(path) | {
                         source: startNode(rel).name,
                         predicate: rel.predicate,
                         target: endNode(rel).name,
                         description: coalesce(rel.description, ''),
                         weight: coalesce(rel.weight, 1.0)
                     }] AS step_dicts,
                     reduce(w = 0.0, r IN relationships(path) | w + coalesce(r.weight, 1.0)) AS total_weight
                RETURN step_dicts, total_weight
                ORDER BY total_weight DESC
                LIMIT $limit
                """,
                {"focals": focal_entities, "limit": limit},
            )
            paths = []
            for r in rows:
                step_dicts = r.get("step_dicts") or []
                paths.append(self.build_path(step_dicts))
            return paths
        except Exception as e:
            logger.warning(f"Multi-hop subgraph traversal failed: {e}")
            return []
