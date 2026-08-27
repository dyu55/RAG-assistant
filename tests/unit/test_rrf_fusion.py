"""Unit tests for Reciprocal Rank Fusion (RRF) hybrid scoring."""

from __future__ import annotations

import pytest

from core.retriever import RetrievedChunk, reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def _create_chunk(
        self, chunk_id: str, text: str = "text", score: float = 0.8
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=score,
            metadata={"source": "test"},
        )

    def test_empty_input_returns_empty_list(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_ranked_list_preserves_order(self):
        c1 = self._create_chunk("c1", "first", 0.9)
        c2 = self._create_chunk("c2", "second", 0.7)
        c3 = self._create_chunk("c3", "third", 0.5)

        fused = reciprocal_rank_fusion([[c1, c2, c3]], k=60)
        assert len(fused) == 3
        assert [c.chunk_id for c in fused] == ["c1", "c2", "c3"]
        # Score calculation: 1.0 / (60 + 1) = 0.01639
        assert fused[0].score == pytest.approx(1.0 / 61, abs=1e-4)
        assert fused[1].score == pytest.approx(1.0 / 62, abs=1e-4)
        assert fused[2].score == pytest.approx(1.0 / 63, abs=1e-4)

    def test_consensus_boosts_common_item(self):
        # c2 appears in BOTH list 1 (rank 2) and list 2 (rank 1)
        # c1 appears only in list 1 (rank 1)
        # c3 appears only in list 2 (rank 2)
        c1 = self._create_chunk("c1", "chunk 1")
        c2_a = self._create_chunk("c2", "chunk 2 (vector)")
        c2_b = self._create_chunk("c2", "chunk 2 (graph)")
        c3 = self._create_chunk("c3", "chunk 3")

        list1 = [c1, c2_a]  # c1: rank 1, c2: rank 2
        list2 = [c2_b, c3]  # c2: rank 1, c3: rank 2

        fused = reciprocal_rank_fusion([list1, list2], k=60)

        # Expected scores:
        # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.03252
        # c1: 1/(60+1) = 0.016393
        # c3: 1/(60+2) = 0.016129
        assert len(fused) == 3
        assert fused[0].chunk_id == "c2"  # Consensus item wins!
        assert fused[1].chunk_id == "c1"
        assert fused[2].chunk_id == "c3"
        assert fused[0].score > fused[1].score > fused[2].score
        assert "rrf_score" in fused[0].metadata

    def test_weighted_fusion_prioritizes_higher_weight_channel(self):
        c1 = self._create_chunk("c1")  # Top of list 1 (weight 2.0)
        c2 = self._create_chunk("c2")  # Top of list 2 (weight 1.0)

        # List 1 has weight 2.0, List 2 has weight 1.0
        fused = reciprocal_rank_fusion([[c1], [c2]], weights=[2.0, 1.0], k=60)
        assert len(fused) == 2
        assert fused[0].chunk_id == "c1"
        assert fused[0].score == pytest.approx(2.0 / 61, abs=1e-4)
        assert fused[1].score == pytest.approx(1.0 / 61, abs=1e-4)

    def test_uneven_list_lengths_handled_gracefully(self):
        c1 = self._create_chunk("c1")
        c2 = self._create_chunk("c2")
        c3 = self._create_chunk("c3")

        list1 = [c1, c2, c3]
        list2 = [c2]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 3
        assert fused[0].chunk_id == "c2"  # In both lists
