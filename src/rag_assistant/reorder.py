"""
Attention-Aware Passage Reordering (Lost-in-the-Middle Mitigation).
Ref: Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2024).

LLMs have high attention at the prompt start (primacy) and prompt end (recency),
with severe attention degradation in the middle.
This module reorganizes ranked evidence into an alternating U-curve layout:
[Rank 1, Rank 3, Rank 5, ..., Rank 6, Rank 4, Rank 2]
"""

from __future__ import annotations

from .models import Evidence


def reorder_lost_in_the_middle(evidence_list: list[Evidence]) -> list[Evidence]:
    """Reorder evidence so the most relevant passages appear at the beginning and end

    of the prompt context (mitigating the 'Lost in the Middle' attention degradation).
    """
    if len(evidence_list) <= 2:
        return list(evidence_list)

    reordered: list[Evidence | None] = [None] * len(evidence_list)
    left = 0
    right = len(evidence_list) - 1

    for i, item in enumerate(evidence_list):
        if i % 2 == 0:
            reordered[left] = item
            left += 1
        else:
            reordered[right] = item
            right -= 1

    return [item for item in reordered if item is not None]
