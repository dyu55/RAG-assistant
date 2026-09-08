from __future__ import annotations

import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path

from .models import Chunk

STOP_WORDS = set(
    "a an the is are was were be been to of for in on at as and or with by from how what why when where who which does do can could would should about tell me this that it its we our their your you i all not".split()
)


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.casefold())
    return [word for word in words if word not in STOP_WORDS]


def normalize(text: str) -> str:
    return " ".join(text.split())


def entities(text: str) -> list[str]:
    explicit = re.findall(r"\[\[([^\]\n]{2,60})\]\]", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+.*$", "", text)
    named = re.findall(r"\b(?:[A-Z][a-zA-Z0-9]+(?:[ -][A-Z][a-zA-Z0-9]+){0,2}|[A-Z]{2,})\b", text)
    result = set()
    connectors = {"after", "before", "when", "each", "during", "once", "if", "then", "while"}
    for name in explicit + named:
        parts = normalize(name).casefold().split()
        while parts and parts[0] in STOP_WORDS | connectors:
            parts.pop(0)
        cleaned = " ".join(parts)
        if len(cleaned) > 1:
            result.add(cleaned)
    return sorted(result)[:24]


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.hidden = max(0, self.hidden - 1)
        else:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def load_pages(filename: str, content: bytes) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}:
        raise ValueError("Supported formats: PDF, Markdown, TXT and HTML")
    if suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported")
            if len(reader.pages) > 1000:
                raise ValueError("PDF exceeds 1,000 pages")
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Unable to parse this PDF") from exc
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text documents must use UTF-8 encoding") from exc
        if "\x00" in text:
            raise ValueError("Binary content is not a text document")
        if suffix in {".html", ".htm"}:
            parser = HTMLText()
            parser.feed(text)
            text = "".join(parser.parts)
        pages = [(1, text)]
    if not any(text.strip() for _, text in pages):
        raise ValueError("No readable text found; scanned PDFs require OCR first")
    return pages


def split_document(filename: str, content: bytes, size: int, overlap: int) -> list[Chunk]:
    document_id = hashlib.sha256(filename.encode()).hexdigest()[:24]
    chunks = []
    for page, raw in load_pages(filename, content):
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start + size // 2, end),
                    text.rfind(" ", start + size // 2, end),
                )
                if boundary > start:
                    end = boundary
            fragment = text[start:end]
            if fragment.strip():
                chunk_id = hashlib.sha256(
                    f"{document_id}:{page}:{start}:{fragment}".encode()
                ).hexdigest()[:24]
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        text=fragment,
                        page=page,
                        start=start,
                        end=end,
                        entities=entities(fragment),
                    )
                )
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
