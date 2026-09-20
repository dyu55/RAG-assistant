from __future__ import annotations

import re
from dataclasses import dataclass

from .text import entities, normalize, tokens


@dataclass
class Proposition:
    id: str
    text: str
    entities: list[str]


def extract_propositions(text: str) -> list[Proposition]:
    """Decompose complex text into atomic, self-contained factual statements (Propositions).

    Implements Dense X Retrieval proposition decomposition principles.
    """
    cleaned = normalize(text)
    if not cleaned:
        return []

    raw_sentences = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    propositions: list[Proposition] = []
    prop_count = 0

    for sent in raw_sentences:
        sent = sent.strip().rstrip(".")
        if len(sent) < 10:
            continue

        clauses = re.split(r";\s*|,\s*(?:and|while|whereas|additionally)\s+|\s+as well as\s+", sent)
        lead_match = re.match(
            r"^((?:(?:The|A|An)\s+)?[A-Z][a-zA-Z0-9_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*)*)\s+(.+)$",
            clauses[0],
        )
        lead_subject = lead_match.group(1) if lead_match else ""
        if not lead_subject and clauses:
            tokens_in_lead = clauses[0].split()
            if tokens_in_lead and tokens_in_lead[0][0].isupper():
                lead_subject = tokens_in_lead[0]

        for idx, clause in enumerate(clauses):
            clause = clause.strip()
            if not clause:
                continue

            if idx > 0 and lead_subject and not re.match(r"^[A-Z]", clause):
                clause = f"{lead_subject} {clause}"

            if len(clause) >= 12:
                prop_count += 1
                propositions.append(
                    Proposition(
                        id=f"P{prop_count}",
                        text=clause + ".",
                        entities=entities(clause),
                    )
                )

    return propositions


def evaluate_proposition_overlap(claim_text: str, propositions: list[Proposition]) -> float:
    """Calculate maximum lexical-semantic alignment between a generated claim and atomic propositions."""
    if not propositions or not claim_text.strip():
        return 0.0

    c_tokens = set(tokens(claim_text))
    if not c_tokens:
        return 0.0

    best_score = 0.0
    for prop in propositions:
        p_tokens = set(tokens(prop.text))
        overlap = len(c_tokens & p_tokens) / max(1, len(c_tokens))
        if overlap > best_score:
            best_score = overlap

    return round(min(1.0, best_score), 4)
