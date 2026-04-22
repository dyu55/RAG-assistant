"""
Document Loader.
Loads PDF, Markdown, TXT, and HTML files into a standardized Document format.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import fitz  # PyMuPDF

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A loaded document with its content and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.metadata.get("filename", "unknown")


def load_file(file_path: str | Path, source_name: str | None = None) -> Document:
    """
    Load a document from a file path. Auto-detects format by extension.

    Args:
        file_path: Path to the file.
        source_name: Optional override for the source name in metadata.

    Returns:
        A Document with extracted text content and metadata.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in settings.SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported types: {settings.SUPPORTED_FILE_TYPES}"
        )

    metadata = {
        "filename": source_name or path.name,
        "file_type": suffix,
        "file_path": str(path.absolute()),
        "loaded_at": datetime.now().isoformat(),
    }

    if suffix == ".pdf":
        content, extra_meta = _load_pdf(path)
        metadata.update(extra_meta)
    elif suffix == ".md":
        content = _load_text(path)
        metadata["file_type_label"] = "markdown"
    elif suffix == ".txt":
        content = _load_text(path)
        metadata["file_type_label"] = "plaintext"
    elif suffix == ".html":
        content = _load_html(path)
        metadata["file_type_label"] = "html"
    else:
        content = _load_text(path)

    logger.info(f"Loaded {path.name}: {len(content)} chars")
    return Document(content=content, metadata=metadata)


def load_from_bytes(
    file_bytes: bytes, filename: str, file_type: str
) -> Document:
    """
    Load a document from raw bytes (for Streamlit file uploads).

    Args:
        file_bytes: Raw file content.
        filename: Original filename.
        file_type: File extension (e.g., ".pdf").
    """
    metadata = {
        "filename": filename,
        "file_type": file_type,
        "loaded_at": datetime.now().isoformat(),
    }

    if file_type == ".pdf":
        content, extra_meta = _load_pdf_bytes(file_bytes)
        metadata.update(extra_meta)
    elif file_type in {".md", ".txt"}:
        content = file_bytes.decode("utf-8", errors="replace")
    elif file_type == ".html":
        content = _strip_html_tags(file_bytes.decode("utf-8", errors="replace"))
    else:
        content = file_bytes.decode("utf-8", errors="replace")

    logger.info(f"Loaded {filename} from bytes: {len(content)} chars")
    return Document(content=content, metadata=metadata)


# ── Private Helpers ───────────────────────────────────────────────────────────


def _load_pdf(path: Path) -> tuple[str, dict]:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages), {"page_count": len(pages)}


def _load_pdf_bytes(file_bytes: bytes) -> tuple[str, dict]:
    """Extract text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages), {"page_count": len(pages)}


def _load_text(path: Path) -> str:
    """Load plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_html(path: Path) -> str:
    """Load HTML file and strip tags for plain text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return _strip_html_tags(raw)


def _strip_html_tags(html: str) -> str:
    """Simple HTML tag stripping without external dependencies."""
    # Remove script and style blocks
    clean = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL)
    # Remove tags
    clean = re.sub(r"<[^>]+>", " ", clean)
    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
