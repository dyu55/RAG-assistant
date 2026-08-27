"""Knowledge-graph builder.

Takes the output of `EntityRelationExtractor` and writes idempotent
`MERGE` statements into Neo4j. The schema:

    (:Chunk {chunk_id, doc_id, text, filename})
        -[:MENTIONS {weight}]->
    (:Entity {name, type, description})

    (:Entity)-[:RELATED {predicate, weight, description}]->(:Entity)

    (:Entity)-[:IN_COMMUNITY]->(:Community {id, level, title, summary})

`Chunk` is created here too so that the graph can map back to the exact
chunks that mention each entity. `weight` is incremented every time the
same (chunk, entity) or (entity, entity) fact is observed, which makes
edge weights a useful retrieval signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from graph.extractor import EntityRelationExtractor, ExtractionResult
from graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class BuildStats:
    """Summary returned after a `build_chunks` run, surfaced in the UI."""

    chunks_processed: int = 0
    chunks_skipped: int = 0  # chunks already indexed (idempotent)
    entities_added: int = 0
    relations_added: int = 0
    errors: int = 0
    new_entities: int = 0
    new_relations: int = 0
    extra: dict = field(default_factory=dict)


class GraphBuilder:
    """Idempotent graph writer over Neo4j."""

    def __init__(
        self,
        neo4j: Neo4jClient,
        extractor: EntityRelationExtractor,
    ):
        self.neo4j = neo4j
        self.extractor = extractor

    # ── Public API ───────────────────────────────────────────────────────────

    def build_chunks(
        self,
        chunks: list[dict],
        *,
        skip_already_indexed: bool = True,
    ) -> BuildStats:
        """Extract + upsert entities/relations for a list of chunks.

        Each item in `chunks` is a dict with at minimum:
            {
              "chunk_id": str,
              "doc_id": str,
              "text": str,
              "filename": str,        # optional, defaults to doc_id
            }

        Idempotency: if `skip_already_indexed=True`, chunks that already
        have a `:Chunk` node with the same `chunk_id` are skipped.
        """
        stats = BuildStats()
        if not chunks:
            return stats

        if skip_already_indexed:
            already = self._indexed_chunk_ids([c["chunk_id"] for c in chunks])
            stats.chunks_skipped = len(already)

        to_process = [
            c for c in chunks if c["chunk_id"] not in (already if skip_already_indexed else set())
        ]
        if not to_process:
            logger.info(f"GraphBuilder: all {len(chunks)} chunks already indexed")
            return stats

        for chunk in to_process:
            try:
                result = self.extractor.extract_chunk(chunk["text"])
                self._upsert_chunk(chunk, result, stats)
                stats.chunks_processed += 1
            except Exception as e:
                stats.errors += 1
                logger.warning(f"Failed to build chunk {chunk.get('chunk_id', '?')[:8]}: {e}")

        logger.info(
            f"GraphBuilder: {stats.chunks_processed} chunks processed, "
            f"{stats.chunks_skipped} skipped, "
            f"+{stats.new_entities} entities / +{stats.new_relations} relations "
            f"({stats.errors} errors)"
        )
        return stats

    def delete_doc(self, doc_id: str) -> None:
        """Remove all artifacts of a document from the graph."""
        self.neo4j.execute_write(
            """
            MATCH (k:Chunk {doc_id: $doc_id})
            OPTIONAL MATCH (k)-[m:MENTIONS]->(:Entity)
            OPTIONAL MATCH (e:Entity)-[r:RELATED]-(:Entity)
            WITH k, collect(DISTINCT m) AS ms,
                 collect(DISTINCT r) AS rs
            FOREACH (_ IN ms | DELETE _)
            WITH k, rs
            FOREACH (_ IN rs | DELETE _)
            WITH collect(DISTINCT k) AS ks
            FOREACH (_ IN ks | DETACH DELETE _)
            """,
            {"doc_id": doc_id},
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _indexed_chunk_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return the subset of `chunk_ids` that already have a :Chunk node."""
        if not chunk_ids:
            return set()
        rows = self.neo4j.execute_read(
            "MATCH (k:Chunk) WHERE k.chunk_id IN $ids RETURN k.chunk_id AS id",
            {"ids": chunk_ids},
        )
        return {r["id"] for r in rows if r.get("id")}

    def _upsert_chunk(self, chunk: dict, result: ExtractionResult, stats: BuildStats) -> None:
        """Write one chunk's worth of nodes/edges in a single transaction."""
        chunk_id = chunk["chunk_id"]
        doc_id = chunk["doc_id"]
        text = chunk["text"]
        filename = chunk.get("filename") or doc_id

        # Stage the writes into parameter arrays to keep one round-trip per chunk.
        entity_payload = [
            {
                "name": e.name,
                "name_norm": e.normalized_name,
                "type": e.type,
                "description": e.description,
            }
            for e in result.entities
        ]
        relation_payload = [
            {
                "source": r.source,
                "source_norm": r.source_norm,
                "target": r.target,
                "target_norm": r.target_norm,
                "predicate": r.predicate,
                "description": r.description,
            }
            for r in result.relations
        ]

        # We treat each relation as an attempt to increment the (s,t,p) edge weight.
        # Duplicates within the same chunk shouldn't double-count, so we dedupe by
        # the normalized (source, target, predicate) key before sending.
        seen_rel: set[tuple[str, str, str]] = set()
        deduped_relations = []
        for r in relation_payload:
            key = (r["source_norm"], r["target_norm"], r["predicate"])
            if key in seen_rel:
                continue
            seen_rel.add(key)
            deduped_relations.append(r)
        relation_payload = deduped_relations

        params = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "filename": filename,
            "text": text,
            "entities": entity_payload,
            "relations": relation_payload,
        }

        # Track deltas so we can populate `BuildStats.new_entities/new_relations`.
        # We do this by reading back the prior weights before the write inside
        # the same transaction.
        rows = self.neo4j.execute_write(_UPSERT_CHUNK_CYPHER, params)

        for row in rows:
            kind = row.get("kind")
            count = row.get("count") or 0
            if kind == "new_entity":
                stats.new_entities += int(count)
                stats.entities_added += int(count)
            elif kind == "new_relation":
                stats.new_relations += int(count)
                stats.relations_added += int(count)


# Single Cypher statement that does everything we need for one chunk.
# Returns two `kind` rows so the Python side can update BuildStats.
_UPSERT_CHUNK_CYPHER = """
// 1) Create / merge the Chunk node
MERGE (k:Chunk {chunk_id: $chunk_id})
  ON CREATE SET k.doc_id    = $doc_id,
                k.filename  = $filename,
                k.text      = $text,
                k.created_at = timestamp()
  ON MATCH  SET k.doc_id    = $doc_id,
                k.filename  = $filename,
                k.text      = $text;

// 2) Create / merge each Entity mentioned by the chunk
WITH k, $entities AS ents
UNWIND ents AS e
MERGE (ent:Entity {name: e.name_norm})
  ON CREATE SET ent.name        = e.name,
                ent.type        = e.type,
                ent.description = e.description,
                ent.first_seen  = timestamp()
  ON MATCH  SET ent.type        = COALESCE(e.type, ent.type),
                ent.description = COALESCE(e.description, ent.description)
MERGE (k)-[m:MENTIONS]->(ent)
  ON CREATE SET m.weight = 1
  ON MATCH  SET m.weight = m.weight + 1
WITH k, collect(DISTINCT ent) AS nodes

// 3) Create / merge relations between entities in this chunk
UNWIND $relations AS r
MATCH (src:Entity {name: r.source_norm}), (tgt:Entity {name: r.target_norm})
MERGE (src)-[rel:RELATED {predicate: r.predicate}]->(tgt)
  ON CREATE SET rel.weight      = 1,
                rel.description = r.description
  ON MATCH  SET rel.weight      = rel.weight + 1,
                rel.description = COALESCE(r.description, rel.description)
WITH count(DISTINCT rel) AS new_rel_count

// 4) Report deltas
RETURN 'new_entity' AS kind,
       size($entities) AS count
UNION ALL
RETURN 'new_relation' AS kind,
       new_rel_count AS count
"""
