"""
Base provider interface for LLM and embedding backends.

Concrete providers should keep network/client-specific behavior in their own
modules while exposing this small surface to the RAG pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    """Common interface for text generation, JSON generation, and embeddings."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.3,
    ) -> str:
        """Generate a plain text response."""

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Always respond with valid JSON.",
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Generate a structured JSON response."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of texts."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier used by the provider."""
