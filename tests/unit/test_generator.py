"""
Unit tests for core/generator.py
Tests the answer generator with mocked LLM provider.
"""

from __future__ import annotations

from unittest.mock import Mock

from core.generator import Citation, GeneratedAnswer, Generator
from core.retriever import RetrievedChunk

# ── Factories ────────────────────────────────────────────────────────────────────


def make_chunk(chunk_id: str, text: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        metadata={"filename": f"{chunk_id}.txt"},
    )


# ── Test: Generator with Mocks ───────────────────────────────────────────────────


class TestGenerator:
    def _make_generator(self, provider_mock):
        return Generator(provider=provider_mock)

    def test_generate_with_valid_chunks_calls_llm(self):
        provider = Mock()
        provider.generate_json.return_value = {
            "answer": "Python is a programming language [Source 1].",
            "citations": [
                {
                    "source_index": 1,
                    "chunk_id": "chunk-abc",
                    "quote": "Python is a programming language",
                }
            ],
            "self_confidence": 0.85,
            "reasoning": "Directly stated in source.",
        }

        chunks = [
            make_chunk("chunk-abc", "Python is a programming language created by Guido.", 0.95)
        ]

        gen = self._make_generator(provider)
        result = gen.generate("What is Python?", chunks)

        assert isinstance(result, GeneratedAnswer)
        assert "Python is a programming language" in result.answer
        assert len(result.citations) == 1
        assert result.self_confidence == 0.85

    def test_generate_empty_chunks_returns_no_context_answer(self):
        provider = Mock()
        gen = self._make_generator(provider)

        result = gen.generate("What is Python?", [])

        assert "no relevant documents" in result.answer.lower()
        assert result.self_confidence == 0.0
        # LLM should NOT be called
        provider.generate_json.assert_not_called()

    def test_generate_llm_failure_returns_error_answer(self):
        provider = Mock()
        provider.generate_json.side_effect = Exception("OpenAI API error: rate limited")

        chunks = [make_chunk("c1", "Python is a programming language.")]
        gen = self._make_generator(provider)

        result = gen.generate("What is Python?", chunks)

        assert "error" in result.answer.lower()
        assert result.self_confidence == 0.0

    def test_generate_parses_citations_correctly(self):
        provider = Mock()
        provider.generate_json.return_value = {
            "answer": "Answer with [Source 1] and [Source 2].",
            "citations": [
                {"source_index": 1, "chunk_id": "c1", "quote": "Quote from source 1."},
                {"source_index": 2, "chunk_id": "c2", "quote": "Quote from source 2."},
            ],
            "self_confidence": 0.8,
            "reasoning": "Based on sources.",
        }

        chunks = [
            make_chunk("c1", "Content 1."),
            make_chunk("c2", "Content 2."),
        ]
        gen = self._make_generator(provider)
        result = gen.generate("Question?", chunks)

        assert len(result.citations) == 2
        assert result.citations[0].chunk_id == "c1"
        assert result.citations[1].chunk_id == "c2"

    def test_generate_infers_chunk_id_from_source_index(self):
        """If chunk_id is missing, infer it from source_index."""
        provider = Mock()
        provider.generate_json.return_value = {
            "answer": "Answer [Source 1].",
            "citations": [
                # chunk_id missing but source_index present
                {"source_index": 1, "chunk_id": "", "quote": "Some quote."},
            ],
            "self_confidence": 0.7,
            "reasoning": "",
        }

        chunks = [make_chunk("c1", "The actual source text.", 0.9)]
        gen = self._make_generator(provider)
        result = gen.generate("Question?", chunks)

        # Should infer chunk_id from source_index
        assert result.citations[0].chunk_id == "c1"

    def test_generate_confidence_clamped_to_valid_range(self):
        provider = Mock()
        provider.generate_json.return_value = {
            "answer": "Answer.",
            "citations": [],
            "self_confidence": 1.5,  # Invalid: > 1.0
            "reasoning": "",
        }

        chunks = [make_chunk("c1", "Content.")]
        gen = self._make_generator(provider)
        result = gen.generate("Question?", chunks)

        assert result.self_confidence == 1.0  # Clamped to 1.0

        provider.generate_json.return_value["self_confidence"] = -0.5  # Invalid: < 0.0
        result2 = gen.generate("Question?", chunks)
        assert result2.self_confidence == 0.0  # Clamped to 0.0

    def test_generate_builds_context_with_numbered_sources(self):
        provider = Mock()
        provider.generate_json.return_value = {
            "answer": "Answer.",
            "citations": [],
            "self_confidence": 0.5,
            "reasoning": "",
        }

        chunks = [
            make_chunk("c1", "First source text.", 0.9),
            make_chunk("c2", "Second source text.", 0.8),
        ]
        gen = self._make_generator(provider)
        _ = gen.generate("What is this?", chunks)

        # Verify the prompt passed to the LLM
        call_args = provider.generate_json.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "[V1]" in prompt
        assert "[V2]" in prompt
        assert "First source text" in prompt
        assert "Second source text" in prompt
        assert "chunk_id: c1" in prompt  # Chunk ID included
        assert "chunk_id: c2" in prompt

    def test_generate_includes_chunk_metadata(self):
        """Each source should include its filename and score."""
        provider = Mock()
        provider.generate_json.return_value = {
            "answer": "Answer.",
            "citations": [],
            "self_confidence": 0.5,
            "reasoning": "",
        }

        chunk = RetrievedChunk(
            chunk_id="c1",
            text="Content",
            score=0.95,
            metadata={"filename": "python-guide.pdf"},
        )
        gen = self._make_generator(provider)
        _ = gen.generate("Question?", [chunk])

        call_args = provider.generate_json.call_args
        prompt = call_args.kwargs.get("prompt") or call_args[1].get("prompt")
        assert "python-guide.pdf" in prompt
        assert "0.95" in prompt


class TestGeneratedAnswer:
    def test_has_citations_true_when_present(self):
        answer = GeneratedAnswer(
            answer="Answer.",
            citations=[Citation(source_index=1, chunk_id="c1", quote="Quote.")],
            self_confidence=0.8,
        )
        assert answer.has_citations is True

    def test_has_citations_false_when_empty(self):
        answer = GeneratedAnswer(
            answer="Answer.",
            citations=[],
            self_confidence=0.8,
        )
        assert answer.has_citations is False

    def test_raw_response_preserved(self):
        raw = {"answer": "A", "citations": [], "self_confidence": 0.9, "reasoning": "R"}
        answer = GeneratedAnswer(
            answer="A",
            citations=[],
            self_confidence=0.9,
            reasoning="R",
            raw_response=raw,
        )
        assert answer.raw_response == raw
