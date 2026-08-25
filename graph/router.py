"""Query router for hybrid GraphRAG + vector RAG.

Decides for every incoming query which path(s) to take:

- `LOCAL`  — answer is about specific entities/concepts; vector + graph
  local retrieval.
- `GLOBAL` — answer requires synthesis across the whole corpus; map-reduce
  over community reports.
- `BOTH`   — combine local + global sources (default for "compare / contrast"
  questions).
- `OFF`    — graph is unavailable; fall back to vector only.

The router is a single LLM call that returns a JSON classification. We
also expose `resolve(mode_setting, llm_decision)` so the UI's explicit
mode selector can short-circuit the LLM.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from providers.base import Provider

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = """You classify a user's question into exactly ONE of these
modes so the system can pick the right retrieval path:

- "LOCAL":  answer is grounded in specific entities or text passages
  ("what does X say about Y", "find the bug in file Z", "who is Alice").
- "GLOBAL": answer requires synthesis across the whole corpus
  ("summarize the main themes", "what are the key findings", "give me
  the high-level overview", "what patterns emerge").
- "BOTH":   answer needs both entity-specific evidence and broad
  synthesis ("compare X and Y across all documents", "how does X relate
  to the overall theme", "summarize X and show specific examples").

Respond with a JSON object:
{"mode": "LOCAL" | "GLOBAL" | "BOTH", "confidence": 0.0-1.0, "reason": "..."}

Output ONLY the JSON object.
"""


class RouteMode(str, Enum):
    OFF = "off"
    LOCAL = "local"
    GLOBAL = "global"
    BOTH = "both"


@dataclass
class RouteDecision:
    mode: RouteMode
    confidence: float = 1.0
    reason: str = ""
    from_setting: bool = False  # True when the user pinned a mode in the UI

    @property
    def run_vector(self) -> bool:
        return self.mode in (RouteMode.OFF, RouteMode.LOCAL, RouteMode.BOTH)

    @property
    def run_graph_local(self) -> bool:
        return self.mode in (RouteMode.LOCAL, RouteMode.BOTH)

    @property
    def run_graph_global(self) -> bool:
        return self.mode in (RouteMode.GLOBAL, RouteMode.BOTH)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "from_setting": self.from_setting,
            "run_vector": self.run_vector,
            "run_graph_local": self.run_graph_local,
            "run_graph_global": self.run_graph_global,
        }


_VALID_MODES = {m.value for m in RouteMode}


class QueryRouter:
    """LLM-based query classifier with explicit-mode override."""

    def __init__(self, provider: Provider, mode_setting: str = "auto"):
        self.provider = provider
        self.mode_setting = (mode_setting or "auto").lower()

    # ── Public API ───────────────────────────────────────────────────────────

    def route(self, query: str) -> RouteDecision:
        """Return the routing decision for the given query."""
        # 1) Explicit setting wins
        if self.mode_setting in _VALID_MODES and self.mode_setting != "auto":
            return RouteDecision(
                mode=RouteMode(self.mode_setting),
                confidence=1.0,
                reason=f"Pinned by GRAPH_RAG_MODE={self.mode_setting}",
                from_setting=True,
            )

        # 2) Lightweight heuristic: if the LLM is unavailable, fall back.
        if self.provider is None:
            return self._heuristic_route(query)

        # 3) LLM classification
        try:
            raw = self.provider.generate_json(
                prompt=f"Classify this question:\n\n{query}",
                system_prompt=ROUTER_SYSTEM_PROMPT,
                temperature=0.0,
            )
            return self._parse(raw)
        except Exception as e:
            logger.warning(f"Router LLM call failed ({e}); using heuristic")
            return self._heuristic_route(query)

    # ── Heuristic fallback ───────────────────────────────────────────────────

    _GLOBAL_PATTERNS = [
        r"\bsummar(y|ize|ise)\b",
        r"\boverview\b",
        r"\b(main|key|major|primary|common|overall)\b.*\btheme(s|)?\b",
        r"\bpattern(s)?\b",
        r"\bhigh[- ]level\b",
        r"\bacross (all|the)\b",
        r"\bcorpus\b",
        r"\bwhat (do(es)?|did) (the|these|all) (doc(s|uments)?|papers?)\b",
        r"\bin (general|summary)\b",
    ]

    def _heuristic_route(self, query: str) -> RouteDecision:
        q = query.lower()
        if any(re.search(p, q) for p in self._GLOBAL_PATTERNS):
            return RouteDecision(
                mode=RouteMode.GLOBAL,
                confidence=0.5,
                reason="Heuristic match for global-style query",
            )
        return RouteDecision(
            mode=RouteMode.LOCAL,
            confidence=0.5,
            reason="Heuristic default to local-style query",
        )

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse(self, raw: dict) -> RouteDecision:
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except Exception:
                return self._heuristic_route(str(raw))

        mode = str(raw.get("mode") or "LOCAL").strip().lower()
        if mode not in _VALID_MODES or mode == "auto":
            mode = "local"

        try:
            conf = float(raw.get("confidence") or 0.5)
        except (TypeError, ValueError):
            conf = 0.5

        return RouteDecision(
            mode=RouteMode(mode),
            confidence=max(0.0, min(1.0, conf)),
            reason=str(raw.get("reason") or ""),
        )