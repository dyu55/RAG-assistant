from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .models import Answer, Chunk


class Store:
    """One transaction replaces a document, its chunks and its graph memberships."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "knowledge.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT OR IGNORE INTO meta VALUES ('schema', '1'), ('revision', '0');
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL UNIQUE, digest TEXT NOT NULL,
                    bytes INTEGER NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL, text TEXT NOT NULL, page INTEGER NOT NULL,
                    start INTEGER NOT NULL, end INTEGER NOT NULL, vector TEXT NOT NULL,
                    entities TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS answers (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
            """)
            if db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] != "1":
                raise ValueError("Unsupported index schema; select a new RAG_DATA_DIR")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def document(self, filename: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE filename=?", (filename,)).fetchone()
            return dict(row) if row else None

    def replace(
        self, chunks: list[Chunk], digest: str, byte_count: int, embedding_identity: str, limit: int
    ) -> dict:
        first = chunks[0]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            identity = db.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            if identity and identity[0] != embedding_identity:
                raise ValueError(
                    "Embedding model changed; use a new RAG_DATA_DIR and reimport documents"
                )
            existing_vector = db.execute("SELECT vector FROM chunks LIMIT 1").fetchone()
            if existing_vector and len(json.loads(existing_vector[0])) != len(first.vector):
                raise ValueError("Embedding dimensions changed; import into a new data directory")
            count = db.execute(
                "SELECT count(*) FROM chunks WHERE document_id != ?", (first.document_id,)
            ).fetchone()[0]
            if count + len(chunks) > limit:
                raise ValueError(
                    f"Index limit is {limit} chunks; remove documents or increase RAG_MAX_CHUNKS"
                )
            db.execute("DELETE FROM documents WHERE id=?", (first.document_id,))
            now = datetime.now(UTC).isoformat()
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                (first.document_id, first.filename, digest, byte_count, now),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.document_id,
                        c.filename,
                        c.text,
                        c.page,
                        c.start,
                        c.end,
                        json.dumps(c.vector),
                        json.dumps(c.entities),
                    )
                    for c in chunks
                ],
            )
            db.execute("INSERT OR REPLACE INTO meta VALUES ('embedding', ?)", (embedding_identity,))
            db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
        return {
            "id": first.document_id,
            "filename": first.filename,
            "chunks": len(chunks),
            "bytes": byte_count,
            "updated_at": now,
            "unchanged": False,
        }

    def snapshot(self, document_ids: list[str] | None = None) -> tuple[int, str, list[Chunk]]:
        with self.connect() as db:
            db.execute("BEGIN")
            revision = int(db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])
            identity = db.execute("SELECT value FROM meta WHERE key='embedding'").fetchone()
            rows = db.execute("SELECT * FROM chunks ORDER BY filename, page, start").fetchall()
        selected = set(document_ids or [])
        chunks = []
        for row in rows:
            if selected and row["document_id"] not in selected:
                continue
            data = dict(row)
            data["vector"] = json.loads(data["vector"])
            data["entities"] = json.loads(data["entities"])
            chunks.append(Chunk(**data))
        return revision, identity[0] if identity else "", chunks

    def list_documents(self) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute("""
                SELECT d.*, count(c.id) AS chunks FROM documents d
                LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.filename
            """)
            ]

    def delete(self, document_id: str) -> bool:
        with self.connect() as db:
            deleted = db.execute("DELETE FROM documents WHERE id=?", (document_id,)).rowcount
            if deleted:
                db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
                if not db.execute("SELECT 1 FROM documents LIMIT 1").fetchone():
                    db.execute("DELETE FROM meta WHERE key='embedding'")
            return bool(deleted)

    def record(self, answer: Answer):
        with self.connect() as db:
            db.execute(
                "INSERT INTO answers VALUES (?, ?, ?)",
                (answer.id, datetime.now(UTC).isoformat(), answer.model_dump_json()),
            )
            db.execute(
                "DELETE FROM answers WHERE id NOT IN (SELECT id FROM answers ORDER BY created_at DESC LIMIT 500)"
            )

    def history(self, limit: int = 30) -> list[dict]:
        with self.connect() as db:
            return [
                {"created_at": row[0], **json.loads(row[1])}
                for row in db.execute(
                    "SELECT created_at, payload FROM answers ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            ]
