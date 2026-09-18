from __future__ import annotations

import re

from .models import Evidence
from .text import tokens

BOILERPLATE_PATTERNS = (
    r"^(all rights reserved|copyright|privacy policy|terms of service|table of contents)\b",
    r"^(click here|read more|learn more|see page|refer to section)\b",
    r"^#{1,6}\s*$",
)


def split_sentences(text: str) -> list[str]:
    """Split passage into natural sentence units preserving readable spans."""
    raw_sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences = []
    for s in raw_sentences:
        clean = s.strip()
        if clean:
            sentences.append(clean)
    return sentences


def is_boilerplate(sentence: str) -> bool:
    """Check if sentence is navigation cues or administrative boilerplate."""
    lower = sentence.lower()
    return any(re.search(pat, lower) for pat in BOILERPLATE_PATTERNS)


def score_sentence(sentence: str, query_tokens: set[str], query_bigrams: set[str]) -> float:
    """Score a sentence based on lexical overlap, phrase match, and informativeness."""
    if len(sentence) < 15 or is_boilerplate(sentence):
        return 0.0

    s_tokens = set(tokens(sentence))
    if not s_tokens:
        return 0.0

    overlap = len(query_tokens & s_tokens) / max(1, len(query_tokens))

    # Bigram phrase bonus
    s_lower = sentence.lower()
    bigram_matches = sum(1 for bg in query_bigrams if bg in s_lower)
    if overlap == 0 and bigram_matches == 0:
        return 0.0

    phrase_bonus = 0.25 * min(1.0, bigram_matches)

    # Lead position / definition bonus (e.g. contains 'is', 'provides', 'uses')
    definition_bonus = (
        0.05 if re.search(r"\b(is|provides|implements|supports|manages)\b", s_lower) else 0.0
    )

    return min(1.0, round(overlap + phrase_bonus + definition_bonus, 4))


def compress_evidence_passage(
    evidence: Evidence,
    query: str,
    target_ratio: float = 0.65,
) -> Evidence:
    """Extractively compress an Evidence passage by pruning irrelevant sentences

    while strictly preserving exact source quotes for citation verification.
    """
    orig_text = evidence.text
    if len(orig_text) <= 140:
        return evidence.model_copy(update={"compressed_text": orig_text, "compression_ratio": 1.0})

    sentences = split_sentences(orig_text)
    if len(sentences) <= 1:
        return evidence.model_copy(update={"compressed_text": orig_text, "compression_ratio": 1.0})

    q_tokens = set(tokens(query))
    q_words = [w for w in re.findall(r"[a-z0-9_]+", query.lower())]
    q_bigrams = {f"{q_words[i]} {q_words[i + 1]}" for i in range(len(q_words) - 1)}

    scored: list[tuple[int, float, str]] = []
    for idx, s in enumerate(sentences):
        score = score_sentence(s, q_tokens, q_bigrams)
        scored.append((idx, score, s))

    # Sort by score descending to pick top informative sentences within budget
    budget_chars = int(len(orig_text) * max(0.2, min(1.0, target_ratio)))
    scored_by_relevance = sorted(scored, key=lambda row: (-row[1], row[0]))

    chosen_indices: set[int] = set()
    accumulated_chars = 0

    for idx, score, sent in scored_by_relevance:
        if score <= 0.0 and len(chosen_indices) > 0:
            continue
        if accumulated_chars + len(sent) <= budget_chars or len(chosen_indices) == 0:
            chosen_indices.add(idx)
            accumulated_chars += len(sent) + 1

    # Preserve chronological passage flow
    selected_sentences = [sentences[i] for i in sorted(chosen_indices)]
    compressed_text = " ".join(selected_sentences)
    ratio = round(len(compressed_text) / max(1, len(orig_text)), 3)

    return evidence.model_copy(
        update={
            "compressed_text": compressed_text,
            "compression_ratio": ratio,
        }
    )


def compress_evidence_set(
    evidence_list: list[Evidence],
    query: str,
    target_ratio: float = 0.65,
    enabled: bool = True,
) -> tuple[list[Evidence], float]:
    """Compress a list of Evidence passages and calculate overall compression ratio."""
    if not enabled or not evidence_list:
        return evidence_list, 1.0

    compressed_list = [
        compress_evidence_passage(item, query, target_ratio) for item in evidence_list
    ]

    total_orig = sum(len(e.text) for e in compressed_list)
    total_comp = sum(len(e.compressed_text or e.text) for e in compressed_list)
    overall_ratio = round(total_comp / max(1, total_orig), 3)

    return compressed_list, overall_ratio
