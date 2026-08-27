"""Graph retriever.

Two search modes, both exposed as `RetrievedChunk`-shaped objects so they
slot into the existing pipeline without modification:

- `local_search(query, top_k)`: extract entities from the query, do a
  bounded Cypher traversal in the entity graph, and return the matching
  entity/relation *descriptions* plus the provenance chunks that
  mention them.

- `global_search(query)`: pick the top communities whose summaries best
  match the query (via the existing `Embedder`), then run the GraphRAG
  *map-reduce* — for each top community, the LLM produces a partial
  answer from its report, and a final reduce step synthesizes them
  into a single answer.

The returned chunks always carry a `source` field that the generator
and the UI can use to distinguish vector vs graph vs community hits.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from config import settings
from core.retriever import RetrievedChunk
from graph.extractor import EntityRelationExtractor
from graph.neo4j_client import Neo4jClient
from ingestion.embedder import Embedder
from providers.base import Provider

logger = logging.getLogger(__name__)


MAP_PROMPT = """You are an analyst answering a question using ONE community
report from a knowledge base.

Produce a JSON object with these fields:
- "answer": a paragraph (3-6 sentences) directly answering the question
  using ONLY the community report. If the report is insufficient, set
  this to "" and explain in "reasoning".
- "relevance": float 0..1, how useful this report is for the question.
- "reasoning": short explanation of what you used and what you ignored.

COMMUNITY REPORT:
{report}

USER QUESTION: {question}

Respond with a single valid JSON object only.
"""

REDUCE_PROMPT = """You are an analyst producing a final answer from several
partial answers, each derived from a different community report.

Combine the partial answers into a single cohesive response. Cite the
community index in brackets, e.g. [Community 1], [Community 2]. If the
partials are all empty / irrelevant, say so plainly.

PARTIAL ANSWERS:
{partials}

USER QUESTION: {question}

Respond with a JSON object: {{"answer": str, "reasoning": str, "cited": [ints]}}
"""


@dataclass
class GraphHit:
    """Internal representation of a single graph search result."""

    text: str
    score: float
    source: str  # "graph" or "community"
    metadata: dict = field(default_factory=dict)


class GraphRetriever:
    """Search the Neo4j knowledge graph and format hits as RetrievedChunks."""

    def __init__(
        self,
        neo4j: Neo4jClient,
        provider: Provider,
        embedder: Embedder | None = None,
        extractor: EntityRelationExtractor | None = None,
    ):
        self.neo4j = neo4j
        self.provider = provider
        self.embedder = embedder
        # Reuse the entity extractor for query-side entity extraction.
        self.extractor = extractor or EntityRelationExtractor(provider)

    # ── Local search ─────────────────────────────────────────────────────────

    def local_search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Entity-anchored traversal returning subgraph descriptions."""
        top_k = top_k or settings.GRAPH_LOCAL_TOPK
        if not query.strip():
            return []

        # 1) Pull query entities.
        try:
            query_ents = self.extractor.extract_query_entities(query)
        except Exception as e:
            logger.warning(f"Query entity extraction failed: {e}")
            query_ents = []

        query_names = [e.normalized_name for e in query_ents if e.normalized_name]
        if not query_names:
            # Fall back to a text-based entity match using Cypher's `CONTAINS`.
            rows = self._entity_overlap_search(query)
        else:
            rows = self._subgraph_search(query_names)

        chunks: list[RetrievedChunk] = []
        for row in rows:
            text = self._format_subgraph(row)
            score = float(row.get("path_weight") or 0.0)
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"graph:{row.get('focal_entity')}:{uuid.uuid4().hex[:8]}",
                    text=text,
                    score=min(1.0, score / max(1.0, top_k)),
                    metadata={
                        "source": "graph",
                        "focal_entity": row.get("focal_entity"),
                        "related_entities": row.get("related_entities", []),
                        "chunk_ids": row.get("chunk_ids", []),
                        "filenames": row.get("filenames", []),
                    },
                )
            )
            if len(chunks) >= top_k:
                break

        return chunks

    def _subgraph_search(self, query_names: list[str]) -> list[dict]:
        """Cypher: traverse from query entities up to N hops; aggregate weights."""
        rows = self.neo4j.execute_read(
            """
            UNWIND $names AS qname
            MATCH (q:Entity {name: qname})
            OPTIONAL MATCH p = (q)-[r:RELATED*1..2]-(:Entity)
            WITH q, p,
                 reduce(w = 0.0, rel IN relationships(p) | w + coalesce(rel.weight, 1.0)) AS path_weight,
                 [n IN nodes(p) WHERE n:Entity | n.name] AS path_entities,
                 [k IN nodes(p) WHERE k:Chunk | k.chunk_id] AS chunk_ids,
                 [k IN nodes(p) WHERE k:Chunk | k.filename] AS filenames
            RETURN q.name AS focal_entity,
                   path_weight,
                   path_entities,
                   chunk_ids,
                   filenames
            ORDER BY path_weight DESC
            LIMIT $limit
            """,
            {"names": query_names, "limit": settings.GRAPH_LOCAL_TOPK * 3},
        )
        # Dedupe by focal_entity (keep the highest-weight row per focal).
        seen: dict[str, dict] = {}
        for r in rows:
            focal = r.get("focal_entity") or ""
            if focal not in seen or r.get("path_weight", 0) > seen[focal].get("path_weight", 0):
                seen[focal] = {
                    "focal_entity": focal,
                    "path_weight": r.get("path_weight", 0.0),
                    "related_entities": [n for n in (r.get("path_entities") or []) if n != focal],
                    "chunk_ids": list({c for c in (r.get("chunk_ids") or []) if c}),
                    "filenames": list({f for f in (r.get("filenames") or []) if f}),
                }
        return list(seen.values())

    def _entity_overlap_search(self, query: str) -> list[dict]:
        """Fallback when no query entities are extracted: scan Entity names."""
        rows = self.neo4j.execute_read(
            """
            MATCH (e:Entity)-[r:RELATED]->(o:Entity)
            WHERE toLower(e.name) CONTAINS toLower($needle)
               OR toLower(e.description) CONTAINS toLower($needle)
            WITH e, o, r,
                 collect(DISTINCT o.name) AS related_entities,
                 collect(DISTINCT r.predicate) AS predicates
            RETURN e.name AS focal_entity,
                   e.description AS focal_description,
                   size(related_entities) AS path_weight,
                   related_entities,
                   predicates,
                   [] AS chunk_ids,
                   [] AS filenames
            LIMIT $limit
            """,
            {
                "needle": query[:80],
                "limit": settings.GRAPH_LOCAL_TOPK,
            },
        )
        return [
            {
                "focal_entity": r.get("focal_entity"),
                "path_weight": r.get("path_weight") or 0.0,
                "related_entities": r.get("related_entities") or [],
                "predicates": r.get("predicates") or [],
                "chunk_ids": r.get("chunk_ids") or [],
                "filenames": r.get("filenames") or [],
            }
            for r in rows
        ]

    def _format_subgraph(self, row: dict) -> str:
        """Render a subgraph row as natural language for the generator."""
        focal = row.get("focal_entity") or "(unknown)"
        related = row.get("related_entities") or []
        if not related:
            return f"Entity mentioned in the knowledge base: {focal}."
        related_list = ", ".join(related[:10])
        return (
            f"From the knowledge graph, {focal} is connected to "
            f"{related_list}. These relationships were extracted from "
            f"chunks of the uploaded documents."
        )

    # ── Global search (one-shot / map-reduce over community reports) ────────

    def global_search(
        self,
        query: str,
        *,
        one_shot: bool = True,
    ) -> list[RetrievedChunk]:
        """Synthesize answer across top community reports.

        Args:
            query: User's macro/global question.
            one_shot: When True (default), performs fast single-pass synthesis (1 LLM call),
                      reducing latency from ~6s to ~1s. When False, runs full Map-Reduce.
        """
        if not query.strip():
            return []
        top_communities = self._select_communities(query)
        if not top_communities:
            return []

        if one_shot:
            final_text, cited = self._one_shot_step(query, top_communities)
            if not final_text.strip():
                # Fallback to map-reduce if one-shot yields empty
                partials = self._map_step(query, top_communities)
                final_text = self._reduce_step(query, partials)
            else:
                partials = [
                    {
                        "community_id": c["id"],
                        "title": c.get("title", ""),
                        "answer": final_text,
                        "relevance": 1.0,
                    }
                    for c in top_communities
                ]
        else:
            partials = self._map_step(query, top_communities)
            final_text = self._reduce_step(query, partials)
            cited = [
                idx
                for idx, p in enumerate(partials, 1)
                if float(p.get("relevance") or 0.0) > 0.0 and p.get("answer")
            ]

        if not final_text.strip():
            return []

        community_ids = [c["id"] for c in top_communities]
        return [
            RetrievedChunk(
                chunk_id=f"community:{community_ids[0] if community_ids else 'global'}",
                text=final_text,
                score=sum(float(p.get("relevance") or 0.0) for p in partials)
                / max(len(partials), 1),
                metadata={
                    "source": "community",
                    "community_ids": community_ids,
                    "partials": partials,
                    "cited_partials": cited,
                    "one_shot": one_shot,
                },
            )
        ]

    def _one_shot_step(self, query: str, communities: list[dict]) -> tuple[str, list[int]]:
        """Fast single-pass synthesis across community reports (1 LLM call instead of N+1)."""
        reports_block = "\n\n---\n\n".join(
            f"[Community {i}: {c.get('title') or '(untitled)'}]\n{self._format_report(c)}"
            for i, c in enumerate(communities, 1)
        )
        prompt = (
            f"COMMUNITY REPORTS:\n{reports_block}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Synthesize a cohesive answer across the community reports above. "
            f"Cite the relevant communities in brackets like [Community 1], [Community 2].\n"
            f'Respond with a single JSON object: {{"answer": str, "reasoning": str, "cited": [ints]}}'
        )
        try:
            raw = self.provider.generate_json(
                prompt=prompt,
                system_prompt="You are a precise macro-synthesizer. Output valid JSON only.",
                temperature=0.1,
            )
            answer = (raw.get("answer") or "").strip()
            cited = raw.get("cited") or [1]
            return answer, cited if isinstance(cited, list) else [1]
        except Exception as e:
            logger.warning(f"One-shot global synthesis failed ({e}); will attempt fallback")
            return "", []

    def _select_communities(self, query: str) -> list[dict]:
        """Rank communities by query/relevant text similarity."""
        rows = self.neo4j.execute_read(
            """
            MATCH (c:Community)
            WHERE c.summary IS NOT NULL
            RETURN c.id AS id,
                   c.level AS level,
                   c.title AS title,
                   c.summary AS summary,
                   c.findings AS findings,
                   c.key_entities AS key_entities
            """
        )
        if not rows:
            return []
        if self.embedder is None:
            # Without embeddings, fall back to keyword overlap.
            scored = [(self._keyword_score(query, r), r) for r in rows]
        else:
            try:
                scored = self._embedding_score(query, rows)
            except Exception as e:
                logger.warning(f"Community embedding scoring failed: {e}")
                scored = [(self._keyword_score(query, r), r) for r in rows]

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for s, r in scored[: settings.GRAPH_GLOBAL_TOP_COMMUNITIES] if s > 0]
        return top

    def _keyword_score(self, query: str, row: dict) -> float:
        q_tokens = {t for t in query.lower().split() if len(t) > 2}
        if not q_tokens:
            return 0.0
        title = row.get("title") or ""
        summary = row.get("summary") or ""
        findings = " ".join(row.get("findings") or [])
        entities = " ".join(row.get("key_entities") or [])
        text = f"{title}\n{summary}\n{findings}\n{entities}".lower()
        hits = sum(1 for t in q_tokens if t in text)
        return hits / len(q_tokens)

    def _embedding_score(self, query: str, rows: list[dict]) -> list[tuple[float, dict]]:
        """Cosine similarity of (query, community_text) using the Embedder."""
        import math

        texts = [
            f"{(r.get('title') or '').strip()}\n{(r.get('summary') or '').strip()}" for r in rows
        ]
        query_vec = self.embedder.embed_query(query)
        text_vecs = self.embedder._embed_fn(texts)

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        return [(cos(query_vec, v), r) for v, r in zip(text_vecs, rows)]

    def _map_step(self, query: str, communities: list[dict]) -> list[dict]:
        """For each community, ask the LLM for a partial answer + relevance."""
        partials: list[dict] = []
        for c in communities:
            report = self._format_report(c)
            try:
                raw = self.provider.generate_json(
                    prompt=MAP_PROMPT.format(report=report, question=query),
                    system_prompt="You are a precise analyst. Output JSON only.",
                    temperature=0.1,
                )
            except Exception as e:
                logger.warning(f"Map step failed for community {c.get('id')[:8]}: {e}")
                raw = {"answer": "", "relevance": 0.0, "reasoning": str(e)}

            partials.append(
                {
                    "community_id": c.get("id"),
                    "title": c.get("title"),
                    "answer": raw.get("answer", "") or "",
                    "relevance": float(raw.get("relevance") or 0.0),
                    "reasoning": raw.get("reasoning", ""),
                }
            )
        return partials

    def _reduce_step(self, query: str, partials: list[dict]) -> str:
        relevant = [p for p in partials if p.get("answer")]
        if not relevant:
            return ""
        joined = "\n\n".join(
            f"[Community {i}] (relevance={p['relevance']:.2f}) {p['answer']}"
            for i, p in enumerate(relevant, 1)
        )
        try:
            raw = self.provider.generate_json(
                prompt=REDUCE_PROMPT.format(partials=joined, question=query),
                system_prompt="You are a precise synthesizer. Output JSON only.",
                temperature=0.1,
            )
            return (raw.get("answer") or "").strip()
        except Exception as e:
            logger.warning(f"Reduce step failed: {e}")
            return "\n\n".join(p["answer"] for p in relevant if p.get("answer"))

    def _format_report(self, c: dict) -> str:
        findings = c.get("findings") or []
        return (
            f"Title: {c.get('title') or '(untitled)'}\n"
            f"Summary: {c.get('summary') or ''}\n"
            f"Key findings: {'; '.join(findings) if findings else '(none)'}\n"
            f"Key entities: {', '.join(c.get('key_entities') or [])}"
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    def has_graph(self) -> bool:
        """True iff Neo4j is reachable AND has any entities."""
        if not self.neo4j.ping():
            return False
        try:
            rows = self.neo4j.execute_read("MATCH (n:Entity) RETURN count(n) AS c LIMIT 1")
            return bool(rows and rows[0].get("c", 0) > 0)
        except Exception:
            return False
