"""
Embedder.
Generates embeddings and stores them in ChromaDB.
Supports OpenAI API embeddings or local sentence-transformers.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from config import settings
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates embeddings for text chunks and stores them in ChromaDB.

    Supports two backends:
    - "openai": Uses OpenAI text-embedding-3-small (requires API key)
    - "local": Uses sentence-transformers/all-MiniLM-L6-v2 (free, local)
    """

    def __init__(
        self,
        backend: str | None = None,
        openai_provider=None,
    ):
        self.backend = backend or settings.EMBEDDING_BACKEND

        # Initialize ChromaDB persistent client
        db_path = Path(settings.CHROMA_DB_PATH)
        db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))

        # Initialize embedding function based on backend
        if self.backend == "openai":
            if openai_provider is None:
                from providers.openai_provider import OpenAIProvider
                self._openai = OpenAIProvider()
            else:
                self._openai = openai_provider
            self._embed_fn = self._embed_openai
            self._embedding_fn_for_chroma = None
        elif self.backend == "local":
            self._local_model = self._load_local_model()
            self._embed_fn = self._embed_local
            self._embedding_fn_for_chroma = None
        else:
            raise ValueError(f"Unknown embedding backend: {self.backend}")

        logger.info(f"Embedder initialized with backend: {self.backend}")

    def _load_local_model(self):
        """Load the local sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
        logger.info(f"Loaded local embedding model: {settings.LOCAL_EMBEDDING_MODEL}")
        return model

    def get_or_create_collection(self, name: str | None = None) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        collection_name = name or settings.CHROMA_COLLECTION_NAME
        return self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(
        self,
        chunks: list[Chunk],
        collection_name: str | None = None,
        batch_size: int = 64,
    ) -> int:
        """
        Embed chunks and upsert them into ChromaDB.

        Args:
            chunks: List of Chunk objects to embed and store.
            collection_name: ChromaDB collection name.
            batch_size: Number of chunks to process per batch.

        Returns:
            Number of chunks ingested.
        """
        if not chunks:
            logger.warning("No chunks to ingest.")
            return 0

        collection = self.get_or_create_collection(collection_name)

        total_ingested = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "doc_id": c.doc_id,
                    "chunk_index": c.index,
                    "filename": c.metadata.get("filename", "unknown"),
                    "file_type": c.metadata.get("file_type", "unknown"),
                    "char_count": len(c.text),
                }
                for c in batch
            ]

            # Generate embeddings
            embeddings = self._embed_fn(texts)

            # Upsert into ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            total_ingested += len(batch)
            logger.info(
                f"Ingested batch {i // batch_size + 1}: "
                f"{len(batch)} chunks (total: {total_ingested}/{len(chunks)})"
            )

        logger.info(f"Ingestion complete: {total_ingested} chunks into '{collection.name}'")
        return total_ingested

    def get_collection_stats(self, collection_name: str | None = None) -> dict:
        """Get statistics about a ChromaDB collection."""
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()

        # Get unique document IDs
        if count > 0:
            result = collection.get(limit=min(count, 10000), include=["metadatas"])
            doc_ids = set()
            filenames = set()
            for meta in result["metadatas"]:
                doc_ids.add(meta.get("doc_id", "unknown"))
                filenames.add(meta.get("filename", "unknown"))
        else:
            doc_ids = set()
            filenames = set()

        return {
            "collection_name": collection.name,
            "total_chunks": count,
            "total_documents": len(doc_ids),
            "filenames": sorted(filenames),
        }

    # ── Embedding Functions ───────────────────────────────────────────────────

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        return self._openai.embed(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using local sentence-transformers."""
        embeddings = self._local_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Used for retrieval."""
        result = self._embed_fn([query])
        return result[0]
