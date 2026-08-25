"""Graph indexer.

Orchestrates entity/relation extraction, graph build, community
detection, and community summarization. Designed to be called from the
Streamlit ingestion flow with minimal coupling.

Entry points:
- `index_chunks(chunks, *, force_rebuild=False)`: typical incremental
  indexing right after ChromaDB ingestion.
- `rebuild()`: drop the graph and rebuild from scratch — used by the
  "Rebuild graph" sidebar button.
- `summarize()`: re-run the community summarizer (useful when a new
  model is configured, or to refresh stale summaries).

When Neo4j is unreachable or `USE_GRAPH_RAG=false`, every method is a
no-op and returns safely so the vector pipeline keeps working.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

from config import settings
from graph.builder import BuildStats, GraphBuilder
from graph.communities import CommunityDetector, CommunityLevel
from graph.neo4j_client import Neo4jClient, Neo4jUnavailable
from graph.summarizer import CommunitySummarizer, CommunityReport
from ingestion.chunker import Chunk
from providers.base import Provider

logger = logging.getLogger(__name__)


@dataclass
class IndexerReport:
    """Summary of an indexer run."""
    chunks_processed: int = 0
    chunks_skipped: int = 0
    new_entities: int = 0
    new_relations: int = 0
    communities: int = 0
    summarized_communities: int = 0
    elapsed_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_skipped": self.chunks_skipped,
            "new_entities": self.new_entities,
            "new_relations": self.new_relations,
            "communities": self.communities,
            "summarized_communities": self.summarized_communities,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "errors": self.errors,
        }


class GraphIndexer:
    """Build and maintain the Neo4j knowledge graph alongside ChromaDB."""

    def __init__(
        self,
        provider: Provider,
        neo4j: Neo4jClient | None = None,
        *,
        auto_trigger_threshold: int | None = None,
    ):
        self.provider = provider
        self.neo4j = neo4j or Neo4jClient()
        self.threshold = (
            auto_trigger_threshold
            if auto_trigger_threshold is not None
            else settings.GRAPH_REBUILD_THRESHOLD
        )

    # ── Public entry points ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True iff Neo4j is reachable."""
        try:
            return self.neo4j.ping()
        except Exception:
            return False

    def bootstrap(self) -> None:
        """Create constraints/indexes. Safe to call repeatedly."""
        try:
            self.neo4j.bootstrap_schema()
        except Neo4jUnavailable as e:
            logger.warning(f"Graph bootstrap skipped: {e}")

    def index_chunks(self, chunks: Iterable[Chunk | dict]) -> IndexerReport:
        """Incrementally index a list of chunks into the graph.

        Triggers community detection + summarization if the number of
        *new* entities since the last run exceeds `threshold`.
        """
        if not settings.USE_GRAPH_RAG:
            return IndexerReport(extra={"skipped": True, "reason": "USE_GRAPH_RAG=false"})

        start = time.time()
        report = IndexerReport()

        # Normalize input to dicts
        chunk_dicts = [self._normalize_chunk(c) for c in chunks]
        chunk_dicts = [c for c in chunk_dicts if c]
        if not chunk_dicts:
            return report

        if not self.is_available():
            report.errors.append("Neo4j unavailable")
            return report

        self.bootstrap()

        try:
            extractor = self._extractor()
            builder = GraphBuilder(neo4j=self.neo4j, extractor=extractor)
            stats: BuildStats = builder.build_chunks(chunk_dicts)

            report.chunks_processed = stats.chunks_processed
            report.chunks_skipped = stats.chunks_skipped
            report.new_entities = stats.new_entities
            report.new_relations = stats.new_relations
            report.errors.extend([str(e) for e in stats.errors] if isinstance(stats.errors, list) else [])

            # Trigger community detection + summarization only when enough new data.
            if report.new_entities >= self.threshold:
                logger.info(
                    f"GraphIndexer: {report.new_entities} new entities "
                    f">= threshold ({self.threshold}); running communities"
                )
                self._run_communities_and_summarize(report)
            else:
                # Even if we skip new communities, refresh summaries once per
                # call if there are no existing communities at all.
                if self._no_communities():
                    self._run_communities_and_summarize(report)
        except Neo4jUnavailable as e:
            report.errors.append(f"Neo4j unavailable: {e}")
        except Exception as e:
            logger.exception(f"GraphIndexer.index_chunks failed: {e}")
            report.errors.append(str(e))
        finally:
            report.elapsed_ms = round((time.time() - start) * 1000, 1)

        logger.info(f"GraphIndexer: {report.to_dict()}")
        return report

    def rebuild(self) -> IndexerReport:
        """Wipe the graph and start fresh. Returns the resulting report."""
        if not settings.USE_GRAPH_RAG:
            return IndexerReport(extra={"skipped": True, "reason": "USE_GRAPH_RAG=false"})
        if not self.is_available():
            return IndexerReport(errors=["Neo4j unavailable"])

        start = time.time()
        report = IndexerReport()
        try:
            self.neo4j.clear_all()
            self.bootstrap()

            from ingestion.embedder import Embedder
            embedder = Embedder()
            collection = embedder.get_or_create_collection()
            total = collection.count()
            if total == 0:
                report.errors.append("No chunks in ChromaDB to rebuild from")
                return report

            offset = 0
            page = 256
            while offset < total:
                rows = collection.get(
                    limit=page,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = rows.get("ids") or []
                docs = rows.get("documents") or []
                metas = rows.get("metadatas") or []
                batch = []
                for cid, text, meta in zip(ids, docs, metas):
                    if not cid or not text:
                        continue
                    batch.append({
                        "chunk_id": cid,
                        "doc_id": (meta or {}).get("doc_id") or "unknown",
                        "filename": (meta or {}).get("filename") or "unknown",
                        "text": text,
                    })
                if batch:
                    sub = self.index_chunks(batch)
                    report.chunks_processed += sub.chunks_processed
                    report.chunks_skipped += sub.chunks_skipped
                    report.new_entities += sub.new_entities
                    report.new_relations += sub.new_relations
                    report.errors.extend(sub.errors)
                offset += page

            self._run_communities_and_summarize(report)
        except Exception as e:
            logger.exception(f"GraphIndexer.rebuild failed: {e}")
            report.errors.append(str(e))
        finally:
            report.elapsed_ms = round((time.time() - start) * 1000, 1)

        return report

    def summarize(self) -> IndexerReport:
        """Re-run community summarization without re-detecting communities."""
        report = IndexerReport()
        if not settings.USE_GRAPH_RAG or not self.is_available():
            return IndexerReport(errors=["Neo4j unavailable / disabled"])
        try:
            start = time.time()
            rows = self.neo4j.execute_read(
                """
                MATCH (c:Community)
                RETURN c.id AS id, c.level AS level,
                       collect(DISTINCT e.name) AS members
                """
            )
            hierarchy_dict: dict[int, list] = {}
            for r in rows:
                hierarchy_dict.setdefault(int(r.get("level") or 0), []).append(
                    {"id": r.get("id"), "level": int(r.get("level") or 0),
                     "members": list({m for m in (r.get("members") or []) if m})}
                )
            hierarchy = [
                CommunityLevel(level=lv, communities=cs)
                for lv, cs in sorted(hierarchy_dict.items())
            ]
            reports: list[CommunityReport] = CommunitySummarizer(
                self.neo4j, self.provider
            ).summarize_hierarchy(hierarchy)
            report.summarized_communities = len(reports)
            report.elapsed_ms = round((time.time() - start) * 1000, 1)
        except Exception as e:
            report.errors.append(str(e))
        return report

    # ── Internals ────────────────────────────────────────────────────────────

    def _extractor(self):
        from graph.extractor import EntityRelationExtractor
        return EntityRelationExtractor(self.provider)

    def _normalize_chunk(self, c) -> dict | None:
        if c is None:
            return None
        if isinstance(c, Chunk):
            return {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "filename": c.metadata.get("filename") or c.doc_id,
                "text": c.text,
            }
        if isinstance(c, dict):
            text = c.get("text") or ""
            if not text:
                return None
            return {
                "chunk_id": c.get("chunk_id") or c.get("id") or "",
                "doc_id": c.get("doc_id") or "unknown",
                "filename": c.get("filename") or c.get("doc_id") or "unknown",
                "text": text,
            }
        return None

    def _no_communities(self) -> bool:
        try:
            rows = self.neo4j.execute_read(
                "MATCH (c:Community) RETURN count(c) AS c LIMIT 1"
            )
            return not (rows and rows[0].get("c", 0) > 0)
        except Exception:
            return True

    def _run_communities_and_summarize(self, report: IndexerReport) -> None:
        try:
            hierarchy: list[CommunityLevel] = CommunityDetector(
                self.neo4j
            ).detect_and_write()
            total = sum(len(level.communities) for level in hierarchy)
            report.communities = total

            if total > 0:
                summaries = CommunitySummarizer(
                    self.neo4j, self.provider
                ).summarize_hierarchy(hierarchy)
                report.summarized_communities = len(summaries)
        except Exception as e:
            logger.exception(f"Community pass failed: {e}")
            report.errors.append(f"communities: {e}")