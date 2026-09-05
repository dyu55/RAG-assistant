"""
HyDE (Hypothetical Document Embeddings) Query Expansion Module.
Generates hypothetical document answer passages to bridge the lexical & structural
gap between short interrogative user queries and declarative knowledge base passages.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

HYDE_SYSTEM_PROMPT = """You are an expert technical writer and knowledge specialist.
Write a clear, authoritative, and concise hypothetical passage (2-4 sentences) that directly answers the user's question as if it were an excerpt from a verified technical documentation or domain knowledge base.

Rules:
1. Write in a factual, declarative documentation tone.
2. Include key technical terms and concepts relevant to the query.
3. Do NOT include conversational filler like "Here is an excerpt" or "To answer your question".
4. Output ONLY the hypothetical document passage.
"""


class LLMProvider(Protocol):
    """Protocol for LLM providers supporting text generation."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> str: ...


class HyDEGenerator:
    """
    Generates hypothetical document passages for dense retrieval alignment.
    """

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document passage that answers the query.
        Falls back to structured heuristic synthesis if LLM provider is unavailable.
        """
        clean_q = query.strip()
        if not clean_q:
            return ""

        if self.provider is not None:
            try:
                prompt = f"Question: {clean_q}\n\nTechnical Document Passage:"
                doc = self.provider.generate(
                    prompt=prompt,
                    system_prompt=HYDE_SYSTEM_PROMPT,
                    temperature=0.2,
                )
                if doc and doc.strip():
                    return doc.strip()
            except Exception as e:
                logger.warning(
                    f"HyDE generation via provider failed, using heuristic fallback: {e}"
                )

        # Rule-based heuristic pseudo-passage synthesis
        return self._heuristic_hypothetical_doc(clean_q)

    def _heuristic_hypothetical_doc(self, query: str) -> str:
        """Construct a synthetic declarative passage from query keywords."""
        terms = [w for w in re.findall(r"\w+", query) if len(w) > 2]
        keywords = ", ".join(terms[:6]) if terms else query
        return (
            f"This document provides comprehensive technical details regarding {query}. "
            f"Key operational aspects, architectural components, and implementation guidelines "
            f"involving {keywords} are explained with formal definitions and specifications."
        )


default_hyde = HyDEGenerator()


def generate_hyde_passage(query: str, provider: LLMProvider | None = None) -> str:
    """Convenience functional interface for HyDE passage generation."""
    generator = HyDEGenerator(provider=provider) if provider else default_hyde
    return generator.generate_hypothetical_document(query)
