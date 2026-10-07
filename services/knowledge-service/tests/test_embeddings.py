"""OpenRouterEmbedder against a fake OpenRouter (httpx.MockTransport):
batching, ordering, retries, and friendly errors - no key, no cost."""
import json

import httpx
import pytest

from app.embeddings import EmbeddingError, OpenRouterEmbedder


def make(handler, batch_size=2, retries=2):
    return OpenRouterEmbedder("sk-test", "https://openrouter.ai/api/v1", batch_size, 5, retries,
                              "http://x", transport=httpx.MockTransport(handler))


def ok_response(req):
    body = json.loads(req.content)
    data = [{"index": i, "embedding": [float(len(t)), 1.0, 0.0]} for i, t in enumerate(body["input"])]
    return httpx.Response(200, json={"data": list(reversed(data)),          # out of order on purpose
                                     "usage": {"prompt_tokens": 7 * len(body["input"])}})


@pytest.mark.anyio
async def test_batches_and_keeps_order():
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer sk-test"
        assert req.url.path == "/api/v1/embeddings"
        return ok_response(req)
    vecs = await make(handler).embed(["a", "bb", "ccc", "dddd", "eeeee"], "openai/text-embedding-3-small")
    assert [v[0] for v in vecs] == [1, 2, 3, 4, 5]                  # order restored by index
    assert [len(c["input"]) for c in calls] == [2, 2, 1]             # batch_size = 2
    assert calls[0]["model"] == "openai/text-embedding-3-small" and calls[0]["encoding_format"] == "float"


@pytest.mark.anyio
async def test_retries_on_429_then_succeeds(monkeypatch):
    import app.embeddings as e

    async def no_sleep(_):
        return None
    monkeypatch.setattr(e.asyncio, "sleep", no_sleep)
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(429, json={"error": "slow down"}) if n["i"] < 3 else ok_response(req)
    vecs = await make(handler).embed(["x"], "m")
    assert len(vecs) == 1 and n["i"] == 3


@pytest.mark.anyio
@pytest.mark.parametrize("status,needle,http", [(401, "API key", 502), (402, "credits", 402),
                                                (400, "rejected", 502)])
async def test_client_errors_are_not_retried(status, needle, http):
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(status, text="nope")
    with pytest.raises(EmbeddingError) as err:
        await make(handler).embed(["x"], "m")
    assert needle in str(err.value) and err.value.status == http and n["i"] == 1


@pytest.mark.anyio
async def test_gives_up_after_retries(monkeypatch):
    import app.embeddings as e

    async def no_sleep(_):
        return None
    monkeypatch.setattr(e.asyncio, "sleep", no_sleep)
    with pytest.raises(EmbeddingError) as err:
        await make(lambda req: httpx.Response(503, text="down"), retries=2).embed(["x"], "m")
    assert err.value.status == 503
