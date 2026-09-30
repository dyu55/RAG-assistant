from __future__ import annotations

from rag_assistant.models import Evidence
from rag_assistant.reorder import reorder_lost_in_the_middle


def _create_ev(source_id: str, relevance: float) -> Evidence:
    return Evidence(
        source_id=source_id,
        chunk_id=f"c_{source_id}",
        document_id="doc1",
        filename="test.txt",
        text=f"Content for {source_id}",
        page=1,
        start=0,
        end=20,
        relevance=relevance,
        fusion_score=0.1,
        channels=["keyword"],
    )


def test_reorder_empty_and_short():
    assert reorder_lost_in_the_middle([]) == []
    ev1 = _create_ev("S1", 0.9)
    assert reorder_lost_in_the_middle([ev1]) == [ev1]
    ev2 = _create_ev("S2", 0.8)
    assert reorder_lost_in_the_middle([ev1, ev2]) == [ev1, ev2]


def test_reorder_four_items():
    # Ranks: S1 (highest), S2 (2nd), S3 (3rd), S4 (4th)
    e1 = _create_ev("S1", 0.95)
    e2 = _create_ev("S2", 0.85)
    e3 = _create_ev("S3", 0.75)
    e4 = _create_ev("S4", 0.65)
    items = [e1, e2, e3, e4]

    reordered = reorder_lost_in_the_middle(items)
    # Expected U-shape: [S1 (Rank 1), S3 (Rank 3), S4 (Rank 4), S2 (Rank 2)]
    assert [e.source_id for e in reordered] == ["S1", "S3", "S4", "S2"]
    # Top 2 most relevant must be at head and tail
    assert reordered[0].source_id == "S1"
    assert reordered[-1].source_id == "S2"


def test_reorder_five_items():
    items = [_create_ev(f"S{i}", 1.0 - i * 0.1) for i in range(1, 6)]
    reordered = reorder_lost_in_the_middle(items)
    # i=0 -> S1 at pos 0
    # i=1 -> S2 at pos 4
    # i=2 -> S3 at pos 1
    # i=3 -> S4 at pos 3
    # i=4 -> S5 at pos 2
    # Result: [S1, S3, S5, S4, S2]
    assert [e.source_id for e in reordered] == ["S1", "S3", "S5", "S4", "S2"]
    assert reordered[0].source_id == "S1"
    assert reordered[-1].source_id == "S2"
