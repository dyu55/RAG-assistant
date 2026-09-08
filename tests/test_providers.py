import json
import math

import httpx
import pytest

from rag_assistant.config import Settings
from rag_assistant.models import Evidence
from rag_assistant.providers import ModelClient, ProviderError, local_vectors, normalized_vector


def test_local_vectors_are_deterministic_and_normalized():
    a, b = local_vectors(["Redis caches delivery status", "Redis caches delivery status"])
    assert a == b and math.isclose(sum(x * x for x in a), 1)
    assert local_vectors([""])[0] == [0.0] * 384


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_embedding_http_contract(provider):
    def handler(request):
        data = json.loads(request.content)
        assert data["model"] == "embed-model" and data["input"] == ["one", "two"]
        if provider == "ollama":
            assert request.url.path == "/api/embed" and data["truncate"] is False
            return httpx.Response(200, json={"embeddings": [[1, 0], [0, 1]]})
        assert request.url.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]},
        )

    settings = Settings(
        embedding_provider=provider,
        embedding_model="embed-model",
        embedding_url="http://model/v1" if provider == "openai" else "http://model",
        embedding_api_key="test-key",
    )
    assert ModelClient(settings, httpx.MockTransport(handler)).embed(["one", "two"]) == [
        [1, 0],
        [0, 1],
    ]


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_generation_http_contract(provider):
    draft = {
        "claims": [{"text": "Atlas uses Redis.", "source_id": "S1", "quote": "Atlas uses Redis."}]
    }

    def handler(request):
        data = json.loads(request.content)
        assert data["model"] == "chat-model" and data["stream"] is False
        assert "untrusted" in data["messages"][0]["content"]
        if provider == "ollama":
            assert "properties" in data["format"]
            return httpx.Response(200, json={"message": {"content": json.dumps(draft)}})
        assert data["response_format"]["type"] == "json_object"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(draft)}}]})

    client = ModelClient(
        Settings(provider=provider, model="chat-model"), httpx.MockTransport(handler)
    )
    evidence = Evidence(
        source_id="S1",
        chunk_id="1",
        document_id="1",
        filename="a.txt",
        text="Atlas uses Redis.",
        page=1,
        start=0,
        end=17,
        relevance=1,
        fusion_score=0.01,
        channels=["keyword"],
    )
    assert client.generate("What does Atlas use?", [evidence]).claims[0].source_id == "S1"


@pytest.mark.parametrize(
    "body", [{"embeddings": []}, {"embeddings": [[float("nan")]]}, {"embeddings": [[1], [1, 2]]}]
)
def test_embedding_response_shape_is_validated(body):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )
    )
    client = ModelClient(Settings(embedding_provider="ollama", embedding_model="embed"), transport)
    with pytest.raises(ProviderError):
        client.embed(["one"])


@pytest.mark.parametrize("values", [[], [float("inf")], [float("nan")]])
def test_invalid_vectors_are_rejected(values):
    with pytest.raises(ProviderError):
        normalized_vector(values)


def test_provider_error_does_not_expose_response_body():
    client = ModelClient(
        Settings(provider="ollama", model="test"),
        httpx.MockTransport(lambda request: httpx.Response(401, text="private-response-secret")),
    )
    with pytest.raises(ProviderError) as error:
        client.generate("question", [])
    assert "private-response-secret" not in str(error.value)
