from __future__ import annotations

import re

from .models import Chunk


def extract_breadcrumbs(text: str, offset: int) -> list[str]:
    """Extract hierarchical section breadcrumbs preceding the given character offset in text."""
    if offset <= 0:
        return []

    prefix = text[:offset]
    lines = prefix.splitlines()

    # Track headings by level (e.g., 1 for #, 2 for ##)
    stack: list[tuple[int, str]] = []

    for line in lines:
        stripped = line.strip()
        header_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            # Pop headers of equal or deeper level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

    return [title for _, title in stack]


def format_contextual_header(filename: str, breadcrumbs: list[str]) -> str:
    """Format an informative breadcrumb prefix for contextual grounding."""
    if not breadcrumbs:
        return f"[{filename}]"
    return f"[{filename} > {' > '.join(breadcrumbs)}]"


def expand_context_window(all_chunks: list[Chunk], target: Chunk, max_chars: int = 1200) -> str:
    """Small-to-big context expansion: retrieve adjacent sibling chunks from the same document

    and page to provide rich surrounding context without modifying the target chunk text.
    """
    siblings = [
        c for c in all_chunks if c.document_id == target.document_id and c.page == target.page
    ]
    if len(siblings) <= 1:
        return target.text

    siblings.sort(key=lambda c: c.start)
    target_idx = next((i for i, c in enumerate(siblings) if c.id == target.id), None)
    if target_idx is None:
        return target.text

    window_chunks = [siblings[target_idx]]
    current_length = len(target.text)

    # Greedily expand backward and forward while within budget
    prev_idx = target_idx - 1
    next_idx = target_idx + 1

    while (prev_idx >= 0 or next_idx < len(siblings)) and current_length < max_chars:
        added_any = False
        if prev_idx >= 0:
            prev_chunk = siblings[prev_idx]
            if current_length + len(prev_chunk.text) + 2 <= max_chars:
                window_chunks.insert(0, prev_chunk)
                current_length += len(prev_chunk.text) + 2
                added_any = True
            prev_idx -= 1

        if next_idx < len(siblings):
            next_chunk = siblings[next_idx]
            if current_length + len(next_chunk.text) + 2 <= max_chars:
                window_chunks.append(next_chunk)
                current_length += len(next_chunk.text) + 2
                added_any = True
            next_idx += 1

        if not added_any:
            break

    return "\n\n".join(c.text for c in window_chunks)
