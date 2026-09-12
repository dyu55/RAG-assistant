from __future__ import annotations

from rag_assistant.contextual import (
    expand_context_window,
    extract_breadcrumbs,
    format_contextual_header,
)
from rag_assistant.models import Chunk


def test_extract_breadcrumbs_empty_or_zero():
    assert extract_breadcrumbs("", 0) == []
    assert extract_breadcrumbs("# Header", 0) == []
    assert extract_breadcrumbs("No headers at all", 10) == []


def test_extract_breadcrumbs_hierarchical():
    doc = (
        "# System Architecture\n"
        "Introduction to the platform.\n"
        "## Authentication Service\n"
        "The authentication service validates JWTs.\n"
        "### Token Rotation\n"
        "Refresh tokens rotate every 7 days.\n"
        "## Storage Layer\n"
        "PostgreSQL and Redis handle state.\n"
    )

    offset_jwt = doc.find("validates JWTs")
    crumbs_jwt = extract_breadcrumbs(doc, offset_jwt)
    assert crumbs_jwt == ["System Architecture", "Authentication Service"]

    offset_token = doc.find("Refresh tokens")
    crumbs_token = extract_breadcrumbs(doc, offset_token)
    assert crumbs_token == ["System Architecture", "Authentication Service", "Token Rotation"]

    offset_storage = doc.find("PostgreSQL and Redis")
    crumbs_storage = extract_breadcrumbs(doc, offset_storage)
    assert crumbs_storage == ["System Architecture", "Storage Layer"]


def test_format_contextual_header():
    assert format_contextual_header("readme.md", []) == "[readme.md]"
    assert (
        format_contextual_header("arch.md", ["Architecture", "Storage"])
        == "[arch.md > Architecture > Storage]"
    )


def test_expand_context_window_single_chunk():
    c = Chunk(
        id="c1",
        document_id="doc1",
        filename="doc1.md",
        text="Single chunk alone.",
        page=1,
        start=0,
        end=20,
    )
    assert expand_context_window([c], c) == "Single chunk alone."


def test_expand_context_window_siblings():
    c1 = Chunk(
        id="c1",
        document_id="doc1",
        filename="doc1.md",
        text="Introduction paragraph.",
        page=1,
        start=0,
        end=23,
    )
    c2 = Chunk(
        id="c2",
        document_id="doc1",
        filename="doc1.md",
        text="Core implementation details.",
        page=1,
        start=24,
        end=52,
    )
    c3 = Chunk(
        id="c3",
        document_id="doc1",
        filename="doc1.md",
        text="Conclusion and next steps.",
        page=1,
        start=53,
        end=79,
    )
    chunks = [c1, c2, c3]

    window = expand_context_window(chunks, c2, max_chars=200)
    assert "Introduction paragraph." in window
    assert "Core implementation details." in window
    assert "Conclusion and next steps." in window


def test_expand_context_window_budget_constraint():
    c1 = Chunk(
        id="c1",
        document_id="doc1",
        filename="doc1.md",
        text="A" * 100,
        page=1,
        start=0,
        end=100,
    )
    c2 = Chunk(
        id="c2",
        document_id="doc1",
        filename="doc1.md",
        text="B" * 50,
        page=1,
        start=101,
        end=151,
    )
    chunks = [c1, c2]

    # With small budget, only target chunk fits
    window = expand_context_window(chunks, c2, max_chars=60)
    assert window == "B" * 50
