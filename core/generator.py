"""
Answer Generator.
Produces structured, citation-aware answers constrained to retrieved context.
This is NOT freeform generation — the LLM is forced to cite sources and admit uncertainty.

The sources may come from three paths:
- Vector chunks       → cite as `[V N]`
- Graph-walk chunks   → cite as `[G N]`
- Community reports   → cite as `[C N]`
The chunk_id in each citation is the chunk_id from the corresponding source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.retriever import RetrievedChunk
from providers.base import Provider

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a precise, reliable knowledge assistant. Your role is to answer questions ONLY based on the provided context documents.

STRICT RULES:
1. ONLY use information from the provided context to answer. Do NOT use your own knowledge.
2. Cite your sources inline. Each source is numbered with a prefix that tells you its origin:
   - `[V N]` = vector-retrieved document chunk (passage from an uploaded file)
   - `[G N]` = graph traversal chunk (a relationship between entities)
   - `[C N]` = community report chunk (a synthesized summary of a topic)
   Use the exact prefix-letter and number shown in the source header.
3. If the context does not contain enough information to answer the question, you MUST say "I don't have enough evidence to answer this question based on the available documents."
4. NEVER fabricate or hallucinate information that is not in the context.
5. If you are uncertain, express your uncertainty clearly.
6. Keep your answer focused and concise.

You MUST respond with a valid JSON object in this exact format:
{
    "answer": "Your answer text with inline [V N] / [G N] / [C N] citations...",
    "citations": [
        {
            "source_index": 1,
            "chunk_id": "the chunk_id from the source header",
            "quote": "exact or near-exact quote from the source that supports your claim"
        }
    ],
    "self_confidence": 0.85,
    "reasoning": "Brief explanation of how you derived the answer from the sources"
}

CONFIDENCE SCALE:
- 0.9-1.0: Answer is directly and clearly stated in the sources
- 0.7-0.9: Answer is well-supported but requires some inference
- 0.5-0.7: Answer is partially supported, some uncertainty
- 0.0-0.5: Very weak support, mostly uncertain
"""


@dataclass
class Citation:
    """A single citation linking an answer claim to a source chunk."""
    source_index: int
    chunk_id: str
    quote: str
    # Origin of the cited chunk: "vector", "graph", or "community".
    # Populated automatically from the RetrievedChunk.metadata.
    source_type: str = "vector"


@dataclass
class GeneratedAnswer:
    """Structured answer from the LLM with citations and confidence."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    self_confidence: float = 0.0
    reasoning: str = ""
    raw_response: dict = field(default_factory=dict)

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


class Generator:
    """Generates constrained, citation-aware answers using retrieved context."""

    def __init__(self, provider: Provider):
        self.provider = provider

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        temperature: float = 0.3,
    ) -> GeneratedAnswer:
        """
        Generate a grounded answer with citations.

        Args:
            query: The user's question.
            chunks: Retrieved context chunks.
            temperature: LLM temperature (lower = more deterministic).

        Returns:
            A GeneratedAnswer with structured citations and confidence.
        """
        if not chunks:
            return GeneratedAnswer(
                answer="I cannot answer this question because no relevant documents were found. Please upload relevant documents first.",
                self_confidence=0.0,
                reasoning="No context chunks available for retrieval.",
            )

        # Build the context prompt with numbered sources
        context_prompt = self._build_context_prompt(query, chunks)

        try:
            raw = self.provider.generate_json(
                prompt=context_prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature,
            )
            return self._parse_response(raw, chunks)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GeneratedAnswer(
                answer="I encountered an error while generating the answer. Please try again.",
                self_confidence=0.0,
                reasoning=f"Generation error: {str(e)}",
            )

    def _build_context_prompt(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> str:
        """Build the user prompt with numbered source contexts."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_label = self._label_for(chunk, i)
            context_parts.append(
                f"[{source_label}] (chunk_id: {chunk.chunk_id}, file: {chunk.source}, score: {chunk.score})\n"
                f"{chunk.text}"
            )

        context_block = "\n\n---\n\n".join(context_parts)

        return (
            f"CONTEXT DOCUMENTS:\n\n{context_block}\n\n"
            f"---\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Based ONLY on the context documents above, provide your answer as a JSON object."
        )

    @staticmethod
    def _label_for(chunk: RetrievedChunk, index: int) -> str:
        """Return the citation label for this chunk: `V1`, `G3`, `C7`, etc."""
        src = chunk.retrieval_source
        prefix_map = {
            "vector": "V",
            "graph": "G",
            "community": "C",
        }
        return f"{prefix_map.get(src, 'V')}{index}"

    def _parse_response(
        self, raw: dict, chunks: list[RetrievedChunk]
    ) -> GeneratedAnswer:
        """Parse the LLM's JSON response into a GeneratedAnswer."""
        citations = []
        for cit in raw.get("citations", []):
            source_idx = cit.get("source_index", 0)
            chunk_id = cit.get("chunk_id", "")

            # If chunk_id is missing, try to infer from source_index
            if not chunk_id and 1 <= source_idx <= len(chunks):
                chunk_id = chunks[source_idx - 1].chunk_id

            # Look up the source type so downstream UI can render badges.
            source_type = "vector"
            if 1 <= source_idx <= len(chunks):
                source_type = chunks[source_idx - 1].retrieval_source

            citations.append(
                Citation(
                    source_index=source_idx,
                    chunk_id=chunk_id,
                    quote=cit.get("quote", ""),
                    source_type=source_type,
                )
            )

        confidence = raw.get("self_confidence", 0.0)
        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, float(confidence)))

        return GeneratedAnswer(
            answer=raw.get("answer", "No answer generated."),
            citations=citations,
            self_confidence=confidence,
            reasoning=raw.get("reasoning", ""),
            raw_response=raw,
        )
