"""
Shared fixtures for all tests.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def sample_pdf_bytes():
    """Minimal valid PDF bytes (single page, no text)."""
    # This is a minimal valid PDF with one empty page
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << >> "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000058 00000 n\n0000000115 00000 n\n0000000214 00000 n\n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n308\n%EOF"
    )


@pytest.fixture
def sample_jsonl_lines():
    """Sample JSONL log entries for testing logger."""
    return [
        json.dumps(
            {
                "timestamp": "2026-01-01T10:00:00",
                "query": "What is RAG?",
                "answer": "RAG is retrieval-augmented generation.",
                "should_abstain": False,
                "reliability": {"confidence": 0.85, "score": 0.82},
                "total_latency_ms": 1200,
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-01-01T10:01:00",
                "query": "Hello",
                "answer": "Hello! How can I help?",
                "should_abstain": True,
                "reliability": {"confidence": 0.1, "score": 0.1},
                "total_latency_ms": 300,
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-01-01T10:02:00",
                "query": "Tell me about Python",
                "answer": "Python is a programming language.",
                "should_abstain": False,
                "reliability": {"confidence": 0.90, "score": 0.88},
                "total_latency_ms": 950,
            }
        ),
    ]
