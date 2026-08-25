"""Hierarchical community summarization.

For each community detected by `CommunityDetector`, build a short natural
language report (title + summary + findings + key entities) and store it
as properties on the `:Community` node.

We summarize leaves first (level 0) and then summarize higher-level
communities using the summaries of their child communities as input,
matching the GraphRAG paper's "roll up the hierarchy" pattern.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from graph.communities import CommunityLevel
from graph.neo4j_client import Neo4jClient
from providers.base import Provider

logger = logging.getLogger(__name__)


LEAF_SUMMARY_PROMPT = """You are an expert analyst writing concise cluster reports.

You are given a *community* — a cluster of related entities extracted from a
knowledge base, together with the relations between them and the source
chunks that mention them. Write a JSON report with these fields:

- "title": 3-8 word descriptive title
- "summary": 2-4 sentence overview of what this community is about
- "findings": array of 3-5 short bullet-style claims that capture the most
  important facts in the community (each ≤ 20 words)
- "key_entities": array of the 5 most important entity names (verbatim)

Be faithful to the evidence. Do not invent information. Use the same
language as the source text where possible.

Respond with a single valid JSON object only.
"""


ROLLUP_SUMMARY_PROMPT = """You are an expert analyst producing a high-level report for a
group of related sub-communities from a knowledge base.

You are given the titles and summaries of several child communities.
Synthesize them into one JSON report with these fields:

- "title": 3-8 word descriptive title for the parent community
- "summary": 3-5 sentences covering the union of the child themes
- "findings": array of 3-5 short bullets (each ≤ 20 words)
- "key_entities": array of up to 8 entity names

Stay faithful to the child summaries; do not introduce new claims.
Respond with a single valid JSON object only.
"""


@dataclass
class CommunityReport:
    id: str
    level: int
    title: str
    summary: str
    findings: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)


class CommunitySummarizer:
    """Generate LLM-written reports for every community."""

    def __init__(self, neo4j: Neo4jClient, provider: Provider):
        if provider is None:
            raise ValueError("CommunitySummarizer requires a Provider")
        self.neo4j = neo4j
        self.provider = provider

    # ── Public API ───────────────────────────────────────────────────────────

    def summarize_hierarchy(
        self,
        hierarchy: list[CommunityLevel],
    ) -> list[CommunityReport]:
        """Summarize all communities, leaves first, then roll up."""
        reports: list[CommunityReport] = []
        for level_obj in hierarchy:
            if level_obj.level == 0:
                for c in level_obj.communities:
                    reports.append(self._summarize_leaf(c))
            else:
                for c in level_obj.communities:
                    reports.append(self._summarize_parent(c))

        self._persist(reports)
        return reports

    # ── Leaf summarization ───────────────────────────────────────────────────

    def _summarize_leaf(self, community) -> CommunityReport:
        payload = self._load_leaf_evidence(community)
        if not payload:
            # Empty community — produce a placeholder report.
            report = CommunityReport(
                id=community.id,
                level=community.level,
                title="Empty community",
                summary="No entities or relations were observed for this community.",
                findings=[],
                key_entities=[],
            )
            return report

        prompt = (
            "COMMUNITY MEMBERS:\n"
            + ", ".join(sorted({m["name"] for m in payload["members"]})) + "\n\n"
            + "SAMPLE RELATIONS:\n"
            + "\n".join(f"- {r}" for r in payload["relations"][:30]) + "\n\n"
            + "SAMPLE SOURCE CHUNKS (truncated):\n"
            + "\n---\n".join(payload["chunks"][:8])
        )

        try:
            raw = self.provider.generate_json(
                prompt=prompt,
                system_prompt=LEAF_SUMMARY_PROMPT,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"Leaf summary failed for {community.id[:8]}: {e}")
            raw = self._fallback_report(payload)

        return CommunityReport(
            id=community.id,
            level=community.level,
            title=(raw.get("title") or community.id[:24]).strip(),
            summary=(raw.get("summary") or "").strip(),
            findings=[str(f).strip() for f in (raw.get("findings") or [])][:5],
            key_entities=[str(e).strip() for e in (raw.get("key_entities") or [])][:5],
        )

    def _load_leaf_evidence(self, community) -> dict:
        rows = self.neo4j.execute_read(
            """
            UNWIND $members AS mname
            MATCH (e:Entity {name: mname})
            OPTIONAL MATCH (e)-[r:RELATED]->(o:Entity)
              WHERE o.name IN $members
            OPTIONAL MATCH (k:Chunk)-[:MENTIONS]->(e)
            RETURN e.name        AS name,
                   e.type        AS type,
                   e.description AS description,
                   collect(DISTINCT {
                       s: e.name,
                       t: o.name,
                       p: r.predicate,
                       d: r.description
                   }) AS rels,
                   collect(DISTINCT substr(k.text, 0, 400)) AS chunk_previews
            """,
            {"members": community.members},
        )
        members = []
        relations: list[str] = []
        chunks: list[str] = []
        for r in rows:
            members.append({"name": r.get("name"), "type": r.get("type"), "description": r.get("description")})
            for rel in r.get("rels") or []:
                if rel.get("p"):
                    relations.append(f"{rel.get('s')} -[{rel.get('p')}]-> {rel.get('t')}")
            for chunk in r.get("chunk_previews") or []:
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
        return {"members": members, "relations": relations, "chunks": chunks}

    def _fallback_report(self, payload: dict) -> dict:
        names = [m["name"] for m in payload["members"][:5]]
        return {
            "title": f"Cluster of {len(payload['members'])} entities",
            "summary": "This community contains: " + ", ".join(names) + ".",
            "findings": [],
            "key_entities": names,
        }

    # ── Parent (rollup) summarization ────────────────────────────────────────

    def _summarize_parent(self, community) -> CommunityReport:
        children = self._load_child_reports(community)
        if not children:
            return CommunityReport(
                id=community.id,
                level=community.level,
                title=f"Cluster of {len(community.members)} entities",
                summary="Higher-level community without detailed child summaries.",
                findings=[],
                key_entities=list(community.members)[:5],
            )

        prompt = "CHILD COMMUNITIES:\n\n"
        for i, child in enumerate(children, 1):
            prompt += (
                f"{i}. TITLE: {child.get('title', '?')}\n"
                f"   SUMMARY: {child.get('summary', '')}\n"
                f"   FINDINGS: {'; '.join(child.get('findings') or [])}\n\n"
            )

        try:
            raw = self.provider.generate_json(
                prompt=prompt,
                system_prompt=ROLLUP_SUMMARY_PROMPT,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning(f"Rollup summary failed for {community.id[:8]}: {e}")
            raw = {
                "title": children[0].get("title", "Parent community"),
                "summary": children[0].get("summary", ""),
                "findings": [],
                "key_entities": children[0].get("key_entities", []),
            }

        return CommunityReport(
            id=community.id,
            level=community.level,
            title=(raw.get("title") or community.id[:24]).strip(),
            summary=(raw.get("summary") or "").strip(),
            findings=[str(f).strip() for f in (raw.get("findings") or [])][:5],
            key_entities=[str(e).strip() for e in (raw.get("key_entities") or [])][:8],
        )

    def _load_child_reports(self, community) -> list[dict]:
        rows = self.neo4j.execute_read(
            """
            MATCH (c:Community)-[:INCLUDES]->(child:Community)
            WHERE c.id = $cid
            RETURN child.id AS id,
                   child.title AS title,
                   child.summary AS summary,
                   child.findings AS findings,
                   child.key_entities AS key_entities
            """,
            {"cid": community.id},
        )
        return [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "summary": r.get("summary"),
                "findings": r.get("findings") or [],
                "key_entities": r.get("key_entities") or [],
            }
            for r in rows
        ]

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, reports: list[CommunityReport]) -> None:
        if not reports:
            return
        rows = [
            {
                "id": r.id,
                "title": r.title,
                "summary": r.summary,
                "findings": r.findings,
                "key_entities": r.key_entities,
            }
            for r in reports
        ]
        self.neo4j.execute_write(
            """
            UNWIND $rows AS row
            MATCH (c:Community {id: row.id})
            SET c.title        = row.title,
                c.summary      = row.summary,
                c.findings     = row.findings,
                c.key_entities = row.key_entities,
                c.summarized_at = timestamp()
            RETURN count(*) AS _
            """,
            {"rows": rows},
        )
        logger.info(f"CommunitySummarizer: persisted {len(rows)} reports")