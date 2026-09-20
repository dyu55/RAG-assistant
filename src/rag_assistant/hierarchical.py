from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .models import Chunk, Evidence


@dataclass
class ParentChunk:
    id: str
    document_id: str
    filename: str
    page: int
    start: int
    end: int
    text: str
    child_ids: list[str] = field(default_factory=list)


def build_parent_chunks(
    chunks: list[Chunk], max_parent_size: int = 1800
) -> tuple[dict[str, ParentChunk], dict[str, str]]:
    """Group sequential child chunks from the same document and page into coherent

    higher-level parent contexts (Small-to-Big Hierarchical Indexing).
    """
    if not chunks:
        return {}, {}

    parents: dict[str, ParentChunk] = {}
    child_to_parent: dict[str, str] = {}

    # Group by (document_id, page)
    grouped: dict[tuple[str, int], list[Chunk]] = {}
    for c in sorted(chunks, key=lambda x: (x.document_id, x.page, x.start)):
        key = (c.document_id, c.page)
        grouped.setdefault(key, []).append(c)

    for (doc_id, page), group in grouped.items():
        current_children: list[Chunk] = []
        current_len = 0

        for chunk in group:
            if current_children and (current_len + len(chunk.text) > max_parent_size):
                p = _create_parent(current_children, doc_id, page)
                parents[p.id] = p
                for ch in current_children:
                    child_to_parent[ch.id] = p.id
                current_children = [chunk]
                current_len = len(chunk.text)
            else:
                current_children.append(chunk)
                current_len += len(chunk.text)

        if current_children:
            p = _create_parent(current_children, doc_id, page)
            parents[p.id] = p
            for ch in current_children:
                child_to_parent[ch.id] = p.id

    return parents, child_to_parent


def _create_parent(children: list[Chunk], doc_id: str, page: int) -> ParentChunk:
    first = children[0]
    last = children[-1]
    text_spans = [children[0].text]
    for prev, curr in zip(children[:-1], children[1:], strict=False):
        overlap = max(0, prev.end - curr.start)
        if 0 < overlap < len(curr.text):
            text_spans.append(curr.text[overlap:])
        elif overlap == 0:
            text_spans.append(" " + curr.text)
        else:
            text_spans.append(curr.text)

    parent_text = "".join(text_spans).strip()
    parent_id = hashlib.sha256(
        f"parent:{doc_id}:{page}:{first.start}:{last.end}".encode()
    ).hexdigest()[:24]

    return ParentChunk(
        id=parent_id,
        document_id=doc_id,
        filename=first.filename,
        page=page,
        start=first.start,
        end=last.end,
        text=parent_text,
        child_ids=[c.id for c in children],
    )


def rollup_to_parent_context(
    evidence_list: list[Evidence],
    parents: dict[str, ParentChunk],
    child_to_parent: dict[str, str],
) -> list[Evidence]:
    """Small-to-Big rollup: enrich evidence with full parent context while preserving

    the specific child excerpt for citation verification.
    """
    if not evidence_list or not parents or not child_to_parent:
        return evidence_list

    enriched = []
    for item in evidence_list:
        parent_id = child_to_parent.get(item.chunk_id)
        parent = parents.get(parent_id) if parent_id else None

        if parent:
            context_win = f"[Parent Section: {parent.filename} p.{parent.page}]\n{parent.text}"
            cloned = item.model_copy(update={"context_window": context_win})
            enriched.append(cloned)
        else:
            enriched.append(item)

    return enriched
