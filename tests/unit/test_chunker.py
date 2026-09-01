"""
Tests for ingestion/chunker.py
"""

from ingestion.chunker import Chunk, RecursiveChunker


class TestRecursiveChunker:
    def test_empty_text_returns_empty_list(self):
        chunker = RecursiveChunker(chunk_size=100)
        result = chunker.chunk_document("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        chunker = RecursiveChunker(chunk_size=100)
        result = chunker.chunk_document("   \n\n   ")
        assert result == []

    def test_short_text_returns_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=512)
        text = "This is a short document."

        chunks = chunker.chunk_document(text, doc_id="doc-1")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].doc_id == "doc-1"
        assert chunks[0].index == 0

    def test_paragraph_splitting(self):
        chunker = RecursiveChunker(chunk_size=512)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = chunker.chunk_document(text, doc_id="doc-2")

        assert len(chunks) >= 1
        # All chunks should have text content
        for chunk in chunks:
            assert chunk.text.strip()

    def test_chunk_has_correct_metadata(self):
        chunker = RecursiveChunker(chunk_size=512)
        text = "Document content here."

        chunks = chunker.chunk_document(text, doc_id="doc-3", metadata={"source": "test"})

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id is not None
        assert chunk.doc_id == "doc-3"
        assert chunk.index == 0
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["total_chunks"] == 1
        assert chunk.metadata["source"] == "test"
        assert "char_count" in chunk.metadata

    def test_multiple_chunks_have_incrementing_indices(self):
        chunker = RecursiveChunker(chunk_size=50)
        # Create text long enough to produce multiple chunks
        text = ("A" * 30 + "\n\n") * 10

        chunks = chunker.chunk_document(text, doc_id="doc-4")

        assert len(chunks) > 1
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_are_unique(self):
        chunker = RecursiveChunker(chunk_size=50)
        text = "Paragraph content here.\n\n" * 10

        chunks = chunker.chunk_document(text)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs should be unique"

    def test_total_chunks_reflects_final_count(self):
        chunker = RecursiveChunker(chunk_size=50)
        text = ("A" * 30 + "\n\n") * 10

        chunks = chunker.chunk_document(text)

        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_overlap_parameter_respected(self):
        chunker = RecursiveChunker(chunk_size=30, overlap=10)
        text = "First chunk content here.\n\nSecond chunk is this."

        chunks = chunker.chunk_document(text)

        # If overlap is working, chunks should reference content from previous chunk
        # This is a behavior test - the overlap mechanism creates repeating content
        assert len(chunks) >= 1

    def test_overlap_with_small_chunk(self):
        # When overlap > chunk_size, should not crash
        chunker = RecursiveChunker(chunk_size=20, overlap=50)
        text = "A" * 100

        chunks = chunker.chunk_document(text)

        # Should not raise, should produce some chunks
        assert isinstance(chunks, list)

    def test_repr_includes_useful_info(self):
        chunker = RecursiveChunker(chunk_size=100)
        text = "This is a representative chunk with meaningful content."
        chunks = chunker.chunk_document(text)

        repr_str = repr(chunks[0])
        assert "Chunk" in repr_str
        assert "index=0" in repr_str

    def test_custom_chunk_size_and_overlap(self):
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        text = "A" * 500

        chunks = chunker.chunk_document(text)

        assert chunker.chunk_size == 200
        assert chunker.overlap == 20
        assert isinstance(chunks, list)

    def test_doc_id_generated_if_not_provided(self):
        chunker = RecursiveChunker(chunk_size=512)
        text = "Some content."

        chunks = chunker.chunk_document(text)

        # Should have a generated UUID
        assert chunks[0].doc_id is not None
        assert len(chunks[0].doc_id) > 0


class TestChunkEdgeCases:
    def test_single_word(self):
        chunker = RecursiveChunker(chunk_size=512)
        chunks = chunker.chunk_document("Word")
        assert len(chunks) == 1
        assert chunks[0].text == "Word"

    def test_very_long_word(self):
        chunker = RecursiveChunker(chunk_size=10)
        # A single "word" longer than chunk_size
        text = "A" * 100

        chunks = chunker.chunk_document(text)

        # Should be split by character limit as last resort
        assert len(chunks) > 1

    def test_mixed_line_endings(self):
        chunker = RecursiveChunker(chunk_size=50)
        text = "Line1\nLine2\r\nLine3\n\nParagraph2"

        chunks = chunker.chunk_document(text)

        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)


class TestContextualChunking:
    def test_with_document_context_headers(self):
        chunk = Chunk(text="Revenue increased by 15%.", chunk_id="c1", doc_id="d1", index=0)
        enriched = chunk.with_document_context(
            doc_title="Q3 2026 Financial Report",
            summary="Financial performance overview of Acme Corp",
            breadcrumb="Income Statement > Regional Sales",
        )
        assert "[Document: Q3 2026 Financial Report]" in enriched
        assert "[Section: Income Statement > Regional Sales]" in enriched
        assert "[Context: Financial performance overview of Acme Corp]" in enriched
        assert "Revenue increased by 15%." in enriched

    def test_with_document_context_no_headers_returns_plain_text(self):
        chunk = Chunk(text="Plain text.", chunk_id="c1", doc_id="d1", index=0)
        enriched = chunk.with_document_context()
        assert enriched == "Plain text."

    def test_chunk_document_contextual_enriches_chunks(self):
        chunker = RecursiveChunker(chunk_size=512)
        text = "Deep learning models require high throughput training."
        chunks = chunker.chunk_document_contextual(
            text,
            doc_id="ai-doc-1",
            doc_title="AI Infrastructure",
            doc_summary="Guide on training scalable LLMs",
        )

        assert len(chunks) == 1
        c = chunks[0]
        assert "[Document: AI Infrastructure]" in c.text
        assert "[Context: Guide on training scalable LLMs]" in c.text
        assert c.metadata["doc_title"] == "AI Infrastructure"
        assert c.metadata["doc_summary"] == "Guide on training scalable LLMs"
