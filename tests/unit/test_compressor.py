"""
Unit tests for core/compressor.py (Context Compressor & Lost-in-the-Middle Mitigation).
"""

from __future__ import annotations

from core.compressor import ContextCompressor, _sentence_jaccard
from core.retriever import RetrievedChunk


class TestSentenceJaccard:
    def test_identical_sentences_similarity_is_one(self):
        s = "Retrieval augmented generation connects LLMs with external knowledge."
        assert _sentence_jaccard(s, s) == 1.0

    def test_disjoint_sentences_similarity_is_zero(self):
        s1 = "Apples are delicious fruits."
        s2 = "Quantum computing relies on superposition and qubits."
        assert _sentence_jaccard(s1, s2) == 0.0

    def test_partial_overlap(self):
        s1 = "Transformers use self attention mechanisms."
        s2 = "Transformers implement cross attention mechanisms."
        sim = _sentence_jaccard(s1, s2)
        assert 0.4 < sim < 1.0


class TestContextCompressor:
    def _make_chunk(self, chunk_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=score,
            metadata={},
        )

    def test_empty_chunks_compress(self):
        compressor = ContextCompressor()
        chunks, stats = compressor.compress([])
        assert chunks == []
        assert stats.original_chars == 0

    def test_lost_in_the_middle_reordering(self):
        compressor = ContextCompressor()
        c0 = self._make_chunk("c0", "Score 1.0 content", 1.0)
        c1 = self._make_chunk("c1", "Score 0.9 content", 0.9)
        c2 = self._make_chunk("c2", "Score 0.8 content", 0.8)
        c3 = self._make_chunk("c3", "Score 0.7 content", 0.7)
        c4 = self._make_chunk("c4", "Score 0.6 content", 0.6)

        chunks = [c0, c1, c2, c3, c4]
        reordered = compressor.reorder_lost_in_the_middle(chunks)

        assert len(reordered) == 5
        # Top 1 at index 0 (start)
        assert reordered[0].chunk_id == "c0"
        # Top 2 at index -1 (end)
        assert reordered[-1].chunk_id == "c1"
        # Middle has lowest ranks
        assert reordered[1].chunk_id == "c2"
        assert reordered[2].chunk_id == "c4"
        assert reordered[3].chunk_id == "c3"

    def test_cross_chunk_sentence_deduplication(self):
        compressor = ContextCompressor(dedup_similarity_threshold=0.85)
        sent1 = "GraphRAG significantly enhances complex reasoning over knowledge graphs."
        sent2 = "Another unique fact about community reports and summarization."

        c1 = self._make_chunk("c1", f"{sent1} {sent2}")
        # c2 has the duplicate sentence sent1 plus unique content
        c2 = self._make_chunk(
            "c2", f"{sent1} A third unique sentence describing entity resolution."
        )

        compressed, dropped = compressor.deduplicate_sentences([c1, c2])

        assert dropped == 1
        assert sent1 in compressed[0].text
        # In second chunk, sent1 should be removed
        assert sent1 not in compressed[1].text
        assert "A third unique sentence" in compressed[1].text

    def test_full_compress_pipeline_and_stats(self):
        compressor = ContextCompressor()
        c0 = self._make_chunk(
            "c0", "First document sentence with technical explanation. Second sentence here.", 0.9
        )
        c1 = self._make_chunk(
            "c1", "First document sentence with technical explanation. Third unique sentence.", 0.8
        )

        compressed, stats = compressor.compress([c0, c1], enable_reordering=True, enable_dedup=True)
        assert len(compressed) == 2
        assert stats.sentences_dropped >= 1
        assert stats.compression_ratio <= 1.0
