"""
Document Chunker.
Splits documents into overlapping chunks for embedding and retrieval.
Uses recursive splitting: paragraphs → sentences → character limit.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk from a document, ready for embedding."""

    text: str
    chunk_id: str
    doc_id: str
    index: int  # Position within the document
    metadata: dict = field(default_factory=dict)

    def with_graph_context(
        self,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
    ) -> str:
        """Return the text prefixed with structured entity and relationship tags for embedding.

        This enables dense vector search to match multi-entity and relational queries
        at sub-50ms speed without online graph traversal overhead.
        """
        parts = []
        if entities:
            parts.append(f"[Entities: {', '.join(entities[:10])}]")
        if relations:
            parts.append(f"[Relations: {', '.join(relations[:10])}]")
        if parts:
            return f"{' '.join(parts)}\n\n{self.text}"
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(id={self.chunk_id[:8]}..., index={self.index}, len={len(self.text)}, '{preview}...')"


class RecursiveChunker:
    """
    Splits text into overlapping chunks using a recursive strategy.

    Strategy:
    1. Try splitting by double newline (paragraphs)
    2. If a piece is still too large, split by single newline
    3. If still too large, split by sentence ('. ')
    4. Last resort: split by character limit

    Overlap ensures context at chunk boundaries is preserved.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP
        self._separators = ["\n\n", "\n", ". ", " "]

    def chunk_document(
        self,
        text: str,
        doc_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """
        Split document text into overlapping chunks.

        Args:
            text: Full document text.
            doc_id: Unique identifier for the source document.
            metadata: Additional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []

        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}

        # Split text into raw pieces
        raw_pieces = self._recursive_split(text, 0)

        # Merge small pieces and enforce overlap
        merged = self._merge_with_overlap(raw_pieces)

        # Create Chunk objects
        chunks = []
        for i, piece in enumerate(merged):
            chunk = Chunk(
                text=piece.strip(),
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                index=i,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(merged),
                    "char_count": len(piece.strip()),
                },
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked document {doc_id[:8]}...: {len(chunks)} chunks "
            f"(avg {sum(len(c.text) for c in chunks) // max(len(chunks), 1)} chars)"
        )
        return chunks

    def _recursive_split(self, text: str, sep_index: int) -> list[str]:
        """Recursively split text using progressively finer separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if sep_index >= len(self._separators):
            # Last resort: hard split by character limit
            return self._hard_split(text)

        separator = self._separators[sep_index]
        pieces = text.split(separator)

        result = []
        for piece in pieces:
            if not piece.strip():
                continue
            # Re-add the separator (except for space-based splits)
            if separator != " " and separator != ". ":
                piece_with_sep = piece
            elif separator == ". ":
                piece_with_sep = piece + ". "
            else:
                piece_with_sep = piece

            if len(piece_with_sep) <= self.chunk_size:
                result.append(piece_with_sep)
            else:
                # Piece is still too large → recurse with next separator
                result.extend(self._recursive_split(piece_with_sep, sep_index + 1))

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Split text by character limit as a last resort."""
        pieces = []
        step = max(self.chunk_size - self.overlap, 1)
        for i in range(0, len(text), step):
            piece = text[i : i + self.chunk_size]
            if piece.strip():
                pieces.append(piece)
        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces and add overlap between chunks."""
        if not pieces:
            return []

        merged = []
        current = pieces[0]

        for piece in pieces[1:]:
            # If merging keeps us under chunk_size, merge
            if len(current) + len(piece) + 1 <= self.chunk_size:
                current = current + "\n" + piece
            else:
                merged.append(current)
                # Add overlap from end of previous chunk to start of new one
                if self.overlap > 0 and len(current) > self.overlap:
                    overlap_text = current[-self.overlap :]
                    current = overlap_text + "\n" + piece
                else:
                    current = piece

        if current.strip():
            merged.append(current)

        return merged
