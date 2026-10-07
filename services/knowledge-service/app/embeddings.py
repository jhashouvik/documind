"""Embedding providers.

An embedding model maps text to a fixed-length vector so that texts with
similar meaning end up close together (high cosine similarity).

OpenRouterEmbedder calls OpenRouter's OpenAI-compatible endpoint
    POST https://openrouter.ai/api/v1/embeddings
    {"model": "openai/text-embedding-3-small", "input": ["text 1", "text 2", ...]}
and gets back one vector per input plus the tokens used. We call it with plain
httpx (rather than an SDK) so every detail is visible: batching, timeouts,
retries with backoff, and the error codes you will meet in real life.

Vectors from DIFFERENT models are not comparable (different sizes and
different "spaces"), so every model gets its own Qdrant collection and a
question must be embedded with the same model its documents were.
"""
import asyncio
import hashlib
import logging
import math
import re
import time
from typing import Protocol

import httpx

from . import metrics as m
from .logging_setup import log_extra
from .tracing import span

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str], model: str, kind: str = "documents") -> list[list[float]]: ...
    async def close(self) -> None: ...


def _friendly(status: int, body: str) -> str:
    if status == 401:
        return "OpenRouter rejected the API key (401). Check the documind-llm Secret."
    if status == 402:
        return "OpenRouter account has insufficient credits (402)."
    if status in (400, 404):
        return f"Embedding model rejected by OpenRouter ({status}): {body[:200]}"
    if status == 429:
        return "OpenRouter rate limit reached (429)."
    return f"OpenRouter embeddings error {status}: {body[:200]}"


class OpenRouterEmbedder:
    def __init__(self, api_key: str, base_url: str, batch_size: int, timeout_s: float,
                 max_retries: int, app_url: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name = "openrouter"
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=5.0), transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "HTTP-Referer": app_url, "X-Title": "DocuMind"})

    async def _call(self, texts: list[str], model: str) -> tuple[list[list[float]], int]:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._http.post("/embeddings", json={
                    "model": model, "input": texts, "encoding_format": "float"})
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                status, body = 503, f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" not in data:   # some providers return 200 with an error object
                        raise EmbeddingError(_friendly(502, str(data.get("error", data))))
                    rows = sorted(data["data"], key=lambda r: r["index"])
                    usage = data.get("usage") or {}
                    return [r["embedding"] for r in rows], int(usage.get("prompt_tokens")
                                                               or usage.get("total_tokens") or 0)
                status, body = resp.status_code, resp.text
                if status < 500 and status != 429:           # 4xx: retrying will not help
                    raise EmbeddingError(_friendly(status, body), 502 if status != 402 else 402)
            log.warning("embedding call failed, retrying",
                        extra=log_extra(model=model, attempt=attempt + 1, status=status))
            if attempt < self.max_retries:
                await asyncio.sleep(delay)
                delay *= 2                                     # exponential backoff
        raise EmbeddingError(_friendly(status, body), 503)

    async def embed(self, texts: list[str], model: str, kind: str = "documents") -> list[list[float]]:
        vectors: list[list[float]] = []
        start = time.perf_counter()
        with span(f"embeddings {model}", **{"gen_ai.operation.name": "embeddings",
                                            "gen_ai.system": "openrouter",
                                            "gen_ai.request.model": model,
                                            "documind.embed.kind": kind,
                                            "documind.embed.texts": len(texts)}) as s:
            tokens = 0
            for i in range(0, len(texts), self.batch_size):     # bounded request size
                batch, used = await self._call(texts[i:i + self.batch_size], model)
                vectors.extend(batch)
                tokens += used
            s.set_attribute("gen_ai.usage.input_tokens", tokens)
        m.EMBED_LATENCY.labels(kind, model).observe(time.perf_counter() - start)
        m.EMBED_TOKENS.labels(model).inc(tokens)
        return vectors

    async def close(self) -> None:
        await self._http.aclose()


class HashEmbedder:
    """Deterministic 'bag of hashed words' vectors. NOT semantic - it only
    matches shared words. Used by unit tests and CI (no API key, no cost)."""

    def __init__(self, dim: int = 384) -> None:
        self.name = "hash"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed(self, texts: list[str], model: str, kind: str = "documents") -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def close(self) -> None:
        return None


def build_embedder(settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashEmbedder(settings.hash_dim)
    return OpenRouterEmbedder(
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.embeddings_base_url, batch_size=settings.embed_batch_size,
        timeout_s=settings.embed_timeout_s, max_retries=settings.embed_max_retries,
        app_url=settings.app_url)
