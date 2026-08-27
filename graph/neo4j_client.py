"""Neo4j client wrapper.

A thin convenience layer around the official `neo4j` Python driver that:
- reads connection details from settings (with sensible defaults)
- exposes a `ping()` health check that never raises (returns False instead)
- bootstraps constraints / indexes used by GraphRAG
- is safe to import without a running Neo4j (used at module load time)

The driver is created lazily so that simply importing this module never
opens a socket. Use `Neo4jClient()` and call `connect()` (or rely on the
auto-connect on first `execute(...)`) only when you actually need the graph.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from config import settings

logger = logging.getLogger(__name__)


class Neo4jUnavailable(RuntimeError):
    """Raised when a graph operation requires Neo4j but it is not reachable."""


class Neo4jClient:
    """Lazy, env-driven Neo4j client.

    All public methods that talk to the database call `connect()` first, so
    a missing/unreachable Neo4j never crashes import-time code paths.
    """

    # Cypher used to ensure the schema exists. We keep this idempotent so
    # it is safe to call on every startup of the app.
    BOOTSTRAP_CYPHER = [
        # Entity uniqueness
        "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
        # Community uniqueness
        "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
        # Helpful lookup index for community level
        "CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.level)",
        # Helpful index for traversal from a chunk to its entities
        "CREATE INDEX chunk_doc IF NOT EXISTS FOR (k:Chunk) ON (k.doc_id)",
    ]

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE
        self._driver = None
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _import_driver(self):
        """Import the neo4j driver lazily so the package is optional at import time."""
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised in misconfigured envs
            raise Neo4jUnavailable(
                "The `neo4j` Python driver is not installed. Install it with: pip install neo4j"
            ) from e
        return GraphDatabase

    def connect(self) -> None:
        """Open the driver. Idempotent."""
        if self._connected and self._driver is not None:
            return
        GraphDatabase = self._import_driver()
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            # Don't let a single bad call hang the whole Streamlit page
            connection_timeout=5.0,
            max_connection_pool_size=10,
        )
        self._connected = True
        logger.info(f"Neo4j driver connected to {self.uri}")

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            self._driver = None
            self._connected = False

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ── Health & schema ───────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True iff Neo4j responds to `RETURN 1`."""
        try:
            self.connect()
            with self._session() as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except Exception as e:
            logger.warning(f"Neo4j ping failed: {e}")
            return False

    def bootstrap_schema(self) -> None:
        """Create constraints/indexes used by the GraphRAG pipeline. Idempotent."""
        self.connect()
        with self._session() as session:
            for stmt in self.BOOTSTRAP_CYPHER:
                try:
                    session.run(stmt).consume()
                except Exception as e:
                    # `IF NOT EXISTS` is supported on modern Neo4j; on older
                    # versions we tolerate the error and keep going.
                    logger.debug(f"Bootstrap statement skipped: {stmt[:60]}... ({e})")

    def stats(self) -> dict[str, int]:
        """Return small graph stats for the UI. Empty dict on failure."""
        try:
            self.connect()
            with self._session() as session:
                entities = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
                relations = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS c").single()[
                    "c"
                ]
                communities = session.run("MATCH (c:Community) RETURN count(c) AS c").single()["c"]
                chunks = session.run("MATCH (k:Chunk) RETURN count(k) AS c").single()["c"]
            return {
                "entities": int(entities),
                "relations": int(relations),
                "communities": int(communities),
                "chunks": int(chunks),
            }
        except Exception as e:
            logger.warning(f"Neo4j stats failed: {e}")
            return {}

    # ── Execution helpers ─────────────────────────────────────────────────────

    @contextmanager
    def _session(self) -> Iterator[Any]:
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def execute(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Run a single Cypher statement and return all records as dicts."""
        self.connect()
        with self._session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def execute_write(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Run a write Cypher in a managed transaction."""
        self.connect()
        with self._session() as session:
            # `execute_write` handles retries on transient errors.
            return list(
                session.execute_write(
                    lambda tx: [dict(r) for r in tx.run(cypher, parameters or {})]
                )
            )

    def execute_read(self, cypher: str, parameters: dict | None = None) -> list[dict]:
        """Run a read Cypher in a managed transaction."""
        self.connect()
        with self._session() as session:
            return list(
                session.execute_read(lambda tx: [dict(r) for r in tx.run(cypher, parameters or {})])
            )

    # ── Maintenance helpers used by GraphIndexer ──────────────────────────────

    def clear_all(self) -> None:
        """Delete every node and relationship. Used by full rebuilds."""
        self.connect()
        with self._session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n").consume())
        logger.info("Neo4j graph cleared")

    def delete_doc(self, doc_id: str) -> None:
        """Remove all chunks/entities/relations associated with a doc."""
        self.connect()
        with self._session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (k:Chunk {doc_id: $doc_id})
                    OPTIONAL MATCH (k)-[:MENTIONS]->(e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATED]->()
                    OPTIONAL MATCH ()-[r2:RELATED]->(e)
                    WITH k, e, r, r2,
                         collect(DISTINCT k) AS chunks,
                         collect(DISTINCT r)  AS r1,
                         collect(DISTINCT r2) AS r2s
                    FOREACH (c IN chunks | DETACH DELETE c)
                    """,
                    {"doc_id": doc_id},
                ).consume()
            )
