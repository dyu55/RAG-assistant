"""
Custom LLM Provider.
Supports OpenAI-compatible APIs: OpenAI, Anthropic (via OpenAI-compatible gateway),
Groq, Together AI, Cohere, Fireworks AI, and any other OpenAI-compatible endpoint.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from config import settings
from providers.base import Provider

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 3

# Base URLs for known providers (no trailing slash)
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "cohere": "https://api.cohere.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "azure": "",  # Azure uses a different auth mechanism; handled separately
}


# ── Known model lists ───────────────────────────────────────────────────────────
# These are the most popular models per provider. Users can always enter a custom
# model name if theirs isn't listed.

PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "together": [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ],
    "cohere": [
        "command-r-plus",
        "command-r",
        "command",
        "command-light",
    ],
    "fireworks": [
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/qwen2p5-72b-instruct",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-coder",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
    ],
    "custom": [],  # User provides their own base URL
}


class CustomProvider(Provider):
    """
    OpenAI-compatible LLM provider with multi-provider support.

    Usage:
        # With a preset provider
        provider = CustomProvider(provider="groq", api_key="gsk_...")

        # With a custom OpenAI-compatible endpoint
        provider = CustomProvider(
            provider="custom",
            api_key="...",
            base_url="https://your-vllm-server.com/v1",
            model="meta-llama/Llama-3-70b",
        )
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self._user_base_url = base_url  # Set only for "custom" provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or self._default_model()
        self.embedding_model = settings.EMBEDDING_MODEL

        if not self.api_key:
            raise ValueError("API key is required.")

        # Build base URL
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider in PROVIDER_BASE_URLS:
            self.base_url = PROVIDER_BASE_URLS[provider]
        else:
            self.base_url = PROVIDER_BASE_URLS["openai"]

        # For Anthropic: inject the API key as a header (Anthropic uses
        # x-api-key instead of Authorization: Bearer)
        self._extra_headers: dict[str, str] = {}
        if provider == "anthropic":
            self._extra_headers["x-api-key"] = self.api_key
            # Anthropic does not support the OpenAI /completions endpoint;
            # use chat completions only. Model must be an Anthropic model.
            self.base_url = PROVIDER_BASE_URLS["anthropic"]
        elif provider == "azure":
            raise NotImplementedError(
                "Azure OpenAI is not yet supported. "
                "Use the 'custom' provider with your Azure endpoint URL."
            )

        # Initialize the OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None,
            timeout=DEFAULT_TIMEOUT,
            max_retries=MAX_RETRIES,
            default_headers=self._extra_headers if self._extra_headers else None,
        )

    def _default_model(self) -> str:
        """Return the default model for the selected provider."""
        models = PROVIDER_MODELS.get(self.provider, [])
        return models[0] if models else "gpt-4o-mini"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.3,
    ) -> str:
        """Generate a text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            logger.warning(f"Rate limit hit ({self.provider}): {e}")
            raise
        except APITimeoutError as e:
            logger.warning(f"Request timed out ({self.provider}): {e}")
            raise
        except APIError as e:
            logger.error(f"API error ({self.provider}): {e}")
            raise
        except Exception as e:
            logger.error(f"Generation failed ({self.provider}): {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Always respond with valid JSON.",
        temperature: float = 0.3,
    ) -> dict:
        """Generate a structured JSON response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        except (RateLimitError, APITimeoutError, APIError) as e:
            logger.warning(f"API error during JSON generation ({self.provider}): {e}")
            raise
        except Exception as e:
            logger.error(f"JSON generation failed ({self.provider}): {e}")
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the configured embedding backend.

        For OpenAI-compatible providers that support an embedding endpoint,
        set EMBEDDING_MODEL and EMBEDDING_BACKEND in settings.
        For providers without embedding support, falls back to OpenAI
        (uses OPENAI_API_KEY from environment or passes via api_key).
        """
        # Route to the appropriate embedding backend
        backend = getattr(settings, "EMBEDDING_BACKEND", "openai")

        if backend == "local":
            # Use sentence-transformers — no API key needed
            return self._embed_local(texts)
        else:
            # Use OpenAI-compatible embedding endpoint
            return self._embed_openai_compatible(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embed using sentence-transformers (local, free)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                'Run: pip install sentence-transformers\n'
                "Or set EMBEDDING_BACKEND=openai in your .env."
            )

        model_name = getattr(settings, "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return model.encode(texts, normalize_embeddings=True).tolist()

    def _embed_openai_compatible(self, texts: list[str]) -> list[list[float]]:
        """Embed using an OpenAI-compatible /embeddings endpoint."""
        # If the main provider has an embedding endpoint, use it;
        # otherwise fall back to OpenAI's endpoint.
        embed_base_url = getattr(settings, "EMBEDDING_BASE_URL", None)
        embed_api_key = getattr(settings, "OPENAI_API_KEY", self.api_key)
        embed_model = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-small")

        if embed_base_url:
            embed_client = OpenAI(api_key=embed_api_key, base_url=embed_base_url)
        else:
            embed_client = OpenAI(api_key=embed_api_key)

        try:
            response = embed_client.embeddings.create(
                model=embed_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def get_model_name(self) -> str:
        """Return the current model identifier."""
        return self.model

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return self.provider