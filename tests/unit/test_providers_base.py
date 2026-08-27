"""
Tests for the provider interface used by RAG pipeline components.
"""

from __future__ import annotations

from providers.base import Provider
from providers.openai_provider import OpenAIProvider


def test_openai_provider_implements_provider_interface():
    assert issubclass(OpenAIProvider, Provider)
