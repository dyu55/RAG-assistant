"""Entity & relation extraction.

Wraps the existing `Provider.generate_json(...)` call to ask the LLM to
extract (entity, type, description) tuples and (subject, predicate, object)
relations from a chunk of text.

The schema is intentionally strict and small so that gpt-4o-mini (or any
other JSON-mode capable model) produces stable, parseable output. We also
normalize entity names (strip + collapse whitespace + lowercase) so that
the same concept mentioned in different chunks deduplicates into a single
node during the graph build step.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from providers.base import Provider

logger = logging.getLogger(__name__)


# ── System prompt ──────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are an expert information-extraction system.

From each input text chunk, extract:
1. The salient *entities* (people, organizations, products, concepts,
   technologies, locations, dates, events, technical terms).
2. The *relations* between those entities that are *explicitly stated or
   directly implied* by the text.

Rules:
- Output MUST be a valid JSON object matching the schema below. No prose.
- Entity names should be the shortest unambiguous noun phrase
  (e.g. "GraphRAG" not "the GraphRAG approach").
- Entity types should be a short capitalized label
  (e.g. PERSON, ORG, TECHNOLOGY, CONCEPT, LOCATION, DATE, EVENT, METRIC).
- Relation predicates should be short verb phrases in SCREAMING_SNAKE_CASE
  (e.g. USES, PART_OF, INTRODUCED_BY, MEASURED_IN).
- Only include relations that you can ground in the text. Do not invent.
- Skip generic entities ("document", "section", "table") unless they are
  the actual subject of a meaningful relation.
- Limit to at most 15 entities and 20 relations per chunk.

JSON SCHEMA:
{
  "entities": [
    {"name": "GraphRAG", "type": "TECHNOLOGY",
     "description": "Microsoft's graph-based retrieval-augmented generation method."}
  ],
  "relations": [
    {"source": "GraphRAG", "target": "Microsoft",
     "predicate": "INTRODUCED_BY",
     "description": "GraphRAG was introduced by Microsoft in 2024."}
  ]
}
"""


# ── Datatypes ─────────────────────────────────────────────────────────────────


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""

    @property
    def normalized_name(self) -> str:
        return _normalize(self.name)


@dataclass
class Relation:
    source: str
    target: str
    predicate: str
    description: str = ""

    @property
    def source_norm(self) -> str:
        return _normalize(self.source)

    @property
    def target_norm(self) -> str:
        return _normalize(self.target)


@dataclass
class ExtractionResult:
    """Result of extracting from a single chunk."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"^[\s\W_]+|[\s\W_]+$", re.UNICODE)


def _normalize(name: str) -> str:
    """Canonical form used as the entity key.

    Lowercased, whitespace collapsed, leading/trailing punctuation stripped,
    and `None`/`""` becomes the empty string. This is intentionally
    lossy: "GraphRAG" and "graph rag" collapse to the same key.
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = _WHITESPACE_RE.sub(" ", s)
    s = _PUNCT_STRIP_RE.sub("", s)
    return s


def _strip_markdown_fences(raw: str) -> str:
    """Some non-OpenAI providers wrap JSON in ```json ... ``` fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        # Drop the first line (```json or ```) and the trailing fence.
        lines = raw.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


# ── Main extractor class ──────────────────────────────────────────────────────


class EntityRelationExtractor:
    """Extract entities + relations from text chunks via the existing Provider."""

    MAX_INPUT_CHARS = 4000  # Safety cap to keep prompts small

    def __init__(self, provider: Provider, *, max_retries: int = 2):
        if provider is None:
            raise ValueError("EntityRelationExtractor requires a Provider")
        self.provider = provider
        self.max_retries = max_retries

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_chunk(self, text: str) -> ExtractionResult:
        """Extract entities and relations from a single chunk of text."""
        text = (text or "").strip()
        if not text:
            return ExtractionResult()

        # Truncate very long chunks before prompting
        if len(text) > self.MAX_INPUT_CHARS:
            text = text[: self.MAX_INPUT_CHARS]

        prompt = (
            "Extract entities and relations from the following text. "
            "Return ONLY the JSON object described in the system prompt.\n\n"
            f"TEXT:\n```\n{text}\n```"
        )

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.provider.generate_json(
                    prompt=prompt,
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.0,
                )
                return self._parse(raw)
            except Exception as e:
                last_err = e
                # Backoff before retry
                time.sleep(0.5 * (attempt + 1))
                logger.debug(f"Extraction attempt {attempt + 1} failed: {e}")
        logger.warning(f"Extraction failed after retries: {last_err}")
        return ExtractionResult()

    def extract_query_entities(self, query: str) -> list[Entity]:
        """Extract entities from a user query (used by local graph search)."""
        result = self.extract_chunk(query)
        return result.entities

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse(self, raw) -> ExtractionResult:
        # `Provider.generate_json` already returns a dict, but we are
        # defensive in case someone subclasses and returns a string.
        if isinstance(raw, str):
            raw = json.loads(_strip_markdown_fences(raw))
        if not isinstance(raw, dict):
            raise ValueError(f"Extractor expected dict, got {type(raw).__name__}")

        entities: list[Entity] = []
        seen: set[str] = set()
        for e in raw.get("entities", []) or []:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue
            ent = Entity(
                name=name,
                type=(e.get("type") or "CONCEPT").strip().upper() or "CONCEPT",
                description=(e.get("description") or "").strip(),
            )
            key = ent.normalized_name
            if not key or key in seen:
                continue
            seen.add(key)
            entities.append(ent)

        relations: list[Relation] = []
        seen_rel: set[tuple[str, str, str]] = set()
        for r in raw.get("relations", []) or []:
            if not isinstance(r, dict):
                continue
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            pred = (r.get("predicate") or "").strip().upper()
            if not (src and tgt and pred):
                continue
            rel = Relation(
                source=src,
                target=tgt,
                predicate=pred,
                description=(r.get("description") or "").strip(),
            )
            if not rel.source_norm or not rel.target_norm:
                continue
            key = (rel.source_norm, rel.target_norm, pred)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            relations.append(rel)

        return ExtractionResult(entities=entities, relations=relations, raw=raw)
