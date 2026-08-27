"""Community detection over the entity graph.

Runs Leiden community detection at multiple levels of granularity (a
hierarchy of levels 0..N). Each level produces a partition where each
Entity belongs to exactly one Community. Communities at higher levels
are supersets of communities at lower levels — we represent that via an
`:IN_COMMUNITY {level}` edge and an `:INCLUDES` link from parent to child
communities.

Two backends are supported, in priority order:

1. **Neo4j GDS** (`gds.graph.project` + `gds.leiden.stream`) — preferred
   when the Graph Data Science plugin is available; runs the algorithm
   inside Neo4j.
2. **python-igraph Leiden** — fallback when GDS is not installed. We
   pull the graph into Python, run Leiden, then write the assignments
   back.

Both paths produce the same output: a list of `CommunityLevel` objects.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from config import settings
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A single community at a single level."""

    id: str
    level: int
    members: list[str] = field(default_factory=list)  # normalized entity names


@dataclass
class CommunityLevel:
    """All communities detected at a single level of the hierarchy."""

    level: int
    communities: list[Community] = field(default_factory=list)

    @property
    def num_communities(self) -> int:
        return len(self.communities)


class CommunityDetector:
    """Run Leiden community detection and write results back to Neo4j."""

    GDS_GRAPH_NAME = "rag_graph_entities"

    def __init__(
        self,
        neo4j: Neo4jClient,
        levels: int | None = None,
    ):
        self.neo4j = neo4j
        self.levels = levels or settings.COMMUNITY_LEVELS

    # ── Public API ───────────────────────────────────────────────────────────

    def detect_and_write(self) -> list[CommunityLevel]:
        """Run detection at all levels, persist results, and return them."""
        if self._try_gds():
            try:
                hierarchy = self._run_gds_levels()
                self._persist(hierarchy)
                return hierarchy
            except Exception as e:
                logger.warning(f"GDS Leiden failed ({e}); falling back to python-igraph")

        # Fallback path
        hierarchy = self._run_igraph_levels()
        self._persist(hierarchy)
        return hierarchy

    # ── GDS backend ──────────────────────────────────────────────────────────

    def _try_gds(self) -> bool:
        """True iff the GDS plugin is installed in this Neo4j instance."""
        try:
            rows = self.neo4j.execute_read("RETURN gds.version() AS v")
            return bool(rows and rows[0].get("v"))
        except Exception as e:
            logger.debug(f"GDS not available: {e}")
            return False

    def _run_gds_levels(self) -> list[CommunityLevel]:
        """Run GDS Leiden at multiple `levels` (hierarchy depths)."""
        # 1) Make sure we have a clean graph projection.
        self._gds_project()

        hierarchy: list[CommunityLevel] = []
        for level in range(self.levels):
            rows = self.neo4j.execute_read(
                """
                CALL gds.leiden.stream($graph_name, {
                    relationshipWeightProperty: 'weight',
                    includeIntermediateCommunities: true,
                    maxLevels: $max_levels
                })
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).name AS name,
                       communityId AS communityId,
                       intermediateCommunityIds AS levels
                """,
                {
                    "graph_name": self.GDS_GRAPH_NAME,
                    "max_levels": self.levels,
                },
            )

            # `communityId` at the current level is the leaf assignment;
            # `intermediateCommunityIds[i]` is the assignment at level i.
            by_community: dict[int, list[str]] = {}
            for r in rows:
                cid = int(r["communityId"])
                name = r.get("name")
                if name:
                    by_community.setdefault(cid, []).append(name)

            communities = [
                Community(
                    id=str(cid),
                    level=level,
                    members=sorted(names),
                )
                for cid, names in by_community.items()
            ]
            hierarchy.append(CommunityLevel(level=level, communities=communities))

        self._gds_drop_projection()
        return hierarchy

    def _gds_project(self) -> None:
        """Project the (:Entity)-[:RELATED]->(:Entity) graph for GDS."""
        self._gds_drop_projection()
        self.neo4j.execute_write(
            """
            CALL gds.graph.project(
                $graph_name,
                'Entity',
                {RELATED: {orientation: 'UNDIRECTED', properties: 'weight'}}
            )
            YIELD graphName
            RETURN graphName
            """,
            {"graph_name": self.GDS_GRAPH_NAME},
        )

    def _gds_drop_projection(self) -> None:
        try:
            self.neo4j.execute_write(
                "CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName",
                {"graph_name": self.GDS_GRAPH_NAME},
            )
        except Exception:
            pass  # graph did not exist; ignore

    # ── igraph fallback ──────────────────────────────────────────────────────

    def _run_igraph_levels(self) -> list[CommunityLevel]:
        """Fallback Leiden via python-igraph."""
        try:
            import igraph as ig  # type: ignore
        except ImportError as e:  # pragma: no cover - misconfigured env
            raise RuntimeError(
                "python-igraph is required when GDS is unavailable. "
                "Install with: pip install python-igraph"
            ) from e

        rows = self.neo4j.execute_read(
            """
            MATCH (a:Entity)-[r:RELATED]->(b:Entity)
            RETURN a.name AS a, b.name AS b, coalesce(r.weight, 1.0) AS w
            """
        )
        if not rows:
            return [CommunityLevel(level=0, communities=[])]

        # Build a unique vertex list and edge list
        vertices: list[str] = []
        vertex_index: dict[str, int] = {}
        edges: list[tuple[int, int]] = []
        weights: list[float] = []

        def vidx(name: str) -> int:
            if name not in vertex_index:
                vertex_index[name] = len(vertices)
                vertices.append(name)
            return vertex_index[name]

        for r in rows:
            a = r.get("a")
            b = r.get("b")
            if not a or not b or a == b:
                continue
            ai, bi = vidx(a), vidx(b)
            edges.append((ai, bi))
            weights.append(float(r.get("w") or 1.0))

        g = ig.Graph(n=len(vertices), edges=edges, directed=False)
        g.vs["name"] = vertices
        g.es["weight"] = weights

        # Leiden returns a VertexClustering for the *finest* level;
        # `cluster.graph.community_leiden` with `n=levels` returns a
        # hierarchical clustering tree, but the simplest portable API is
        # to run Leiden at multiple resolutions by varying `resolution_parameter`.
        hierarchy: list[CommunityLevel] = []
        for level in range(self.levels):
            # Resolution grows with level so higher levels produce *fewer*
            # (larger) communities — matching GraphRAG's convention.
            resolution = 1.0 / (1 + level)
            clustering = g.community_leiden(
                weights="weight",
                resolution_parameter=resolution,
            )
            communities: list[Community] = []
            for cid, members in enumerate(clustering):
                if not members:
                    continue
                names = [vertices[i] for i in members]
                communities.append(
                    Community(
                        id=str(cid),
                        level=level,
                        members=sorted(names),
                    )
                )
            hierarchy.append(CommunityLevel(level=level, communities=communities))

        return hierarchy

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, hierarchy: Iterable[CommunityLevel]) -> None:
        """Write communities and IN_COMMUNITY edges to Neo4j (idempotent)."""
        rows = []
        for level_obj in hierarchy:
            for c in level_obj.communities:
                rows.append(
                    {
                        "id": f"L{level_obj.level}-{c.id}-{uuid.uuid4().hex[:6]}",
                        "level": level_obj.level,
                        "members": c.members,
                    }
                )

        if not rows:
            logger.info("CommunityDetector: no communities to write")
            return

        self.neo4j.execute_write(
            """
            // Clear any old community assignments for the levels we're rewriting
            MATCH (c:Community)
            WHERE c.level IN $levels
            DETACH DELETE c
            WITH count(*) AS _

            UNWIND $rows AS row
            MERGE (c:Community {id: row.id})
              ON CREATE SET c.level = row.level
            WITH c, row
            UNWIND row.members AS mname
            MATCH (e:Entity {name: mname})
            MERGE (e)-[r:IN_COMMUNITY]->(c)
              ON CREATE SET r.weight = 1
              ON MATCH  SET r.weight = r.weight + 1
            RETURN count(*) AS _
            """,
            {"rows": rows, "levels": list(range(self.levels))},
        )

        # Build parent/child relations across levels.
        # Level k+1 contains supersets of communities at level k; we link
        # by overlap (a parent contains ≥50% of a child's members).
        self._link_hierarchy(hierarchy)

    def _link_hierarchy(self, hierarchy: list[CommunityLevel]) -> None:
        """Add :INCLUDES edges from higher-level communities to children."""
        if len(hierarchy) < 2:
            return

        for upper_idx in range(1, len(hierarchy)):
            upper = hierarchy[upper_idx]
            lower = hierarchy[upper_idx - 1]
            rows = []
            for parent in upper.communities:
                pset = set(parent.members)
                for child in lower.communities:
                    if not child.members:
                        continue
                    overlap = len(pset.intersection(child.members))
                    if overlap >= 0.5 * len(child.members):
                        rows.append(
                            {
                                "parent": parent.id,
                                "child": child.id,
                                "overlap": overlap,
                            }
                        )
            if not rows:
                continue
            self.neo4j.execute_write(
                """
                UNWIND $rows AS r
                MATCH (p:Community {id: r.parent}), (c:Community {id: r.child})
                MERGE (p)-[rel:INCLUDES]->(c)
                  ON CREATE SET rel.overlap = r.overlap
                RETURN count(*) AS _
                """,
                {"rows": rows},
            )
