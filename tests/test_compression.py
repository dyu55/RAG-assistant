from __future__ import annotations

from rag_assistant.compression import (
    compress_evidence_passage,
    compress_evidence_set,
    is_boilerplate,
    score_sentence,
    split_sentences,
)
from rag_assistant.config import Settings
from rag_assistant.models import Evidence, Question
from rag_assistant.service import KnowledgeService


def test_split_sentences():
    text = "First sentence here. Second sentence starts now! Third sentence ends? Fourth line\n\nFifth."
    sentences = split_sentences(text)
    assert len(sentences) >= 4
    assert "First sentence here." in sentences[0]


def test_is_boilerplate():
    assert is_boilerplate("All rights reserved. Copyright 2026 Acme Corp.") is True
    assert is_boilerplate("Click here to view full table of contents.") is True
    assert is_boilerplate("Atlas uses Redis for distributed session storage.") is False


def test_score_sentence():
    q_tokens = {"atlas", "redis", "cache"}
    q_bigrams = {"atlas redis", "redis cache"}

    relevant_sent = "Atlas uses Redis for high-performance distributed cache."
    score = score_sentence(relevant_sent, q_tokens, q_bigrams)
    assert score > 0.5

    irrelevant_sent = "The weather today is sunny with light clouds."
    assert score_sentence(irrelevant_sent, q_tokens, q_bigrams) == 0.0

    short_sent = "Too short."
    assert score_sentence(short_sent, q_tokens, q_bigrams) == 0.0


def test_compress_evidence_passage():
    long_passage = (
        "Atlas coordinates with Redis cluster for high throughput. "
        "Redis stores session state with automatic failover. "
        "The quick brown fox jumps over the lazy dog repeatedly. "
        "Astronomy is the scientific study of celestial objects and space. "
        "Click here for additional documentation and navigation links. "
        "Atlas provides automatic retry logic when cache is unavailable."
    )
    ev = Evidence(
        source_id="S1",
        chunk_id="c1",
        document_id="d1",
        filename="doc.md",
        text=long_passage,
        page=1,
        start=0,
        end=len(long_passage),
        relevance=0.9,
        fusion_score=0.1,
        channels=["keyword"],
    )

    compressed_ev = compress_evidence_passage(
        ev, query="How does Atlas integrate with Redis cache?", target_ratio=0.60
    )
    assert compressed_ev.compressed_text != ""
    assert compressed_ev.compression_ratio < 1.0
    # Relevant sentences kept
    assert "Atlas coordinates with Redis cluster" in compressed_ev.compressed_text
    # Completely off-topic sentence pruned
    assert "Astronomy is the scientific study" not in compressed_ev.compressed_text
    # All sentences in compressed_text must be exact substrings of original text
    assert compressed_ev.compressed_text.split(". ")[0] in ev.text


def test_compress_evidence_passage_short():
    short_text = "Short passage under threshold."
    ev = Evidence(
        source_id="S1",
        chunk_id="c1",
        document_id="d1",
        filename="doc.md",
        text=short_text,
        page=1,
        start=0,
        end=len(short_text),
        relevance=0.8,
        fusion_score=0.1,
        channels=["keyword"],
    )
    res = compress_evidence_passage(ev, query="Short passage", target_ratio=0.5)
    assert res.compressed_text == short_text
    assert res.compression_ratio == 1.0


def test_compress_evidence_set_disabled():
    ev = Evidence(
        source_id="S1",
        chunk_id="c1",
        document_id="d1",
        filename="doc.md",
        text="Some sample passage text.",
        page=1,
        start=0,
        end=25,
        relevance=0.8,
        fusion_score=0.1,
        channels=["keyword"],
    )
    items, ratio = compress_evidence_set([ev], query="sample", enabled=False)
    assert ratio == 1.0
    assert items[0] == ev


def test_service_context_compression_end_to_end(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        provider="offline",
        context_compression_enabled=True,
        context_compression_ratio=0.65,
    )
    service = KnowledgeService(settings)
    doc_content = (
        b"Atlas coordinates with Redis cluster for high throughput.\n"
        b"Redis stores session state with automatic failover.\n"
        b"Unrelated filler paragraph about weather and gardening tips.\n"
        b"Another irrelevant sentence discussing random sports statistics.\n"
        b"Atlas provides automatic retry logic when cache is unavailable.\n"
    )
    service.ingest("guide.txt", doc_content)

    q = Question(text="How does Atlas integrate with Redis?")
    answer = service.ask(q)

    assert answer.status == "supported"
    assert "compression" in answer.timings_ms
    assert answer.compression_ratio is not None
    assert answer.compression_ratio <= 1.0
    # Strict citation verification must succeed
    assert len(answer.claims) >= 1
    for claim in answer.claims:
        assert any(claim.quote in e.text for e in answer.evidence)
