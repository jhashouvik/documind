"""HTTP client for knowledge-service (service-to-service call inside the cluster).

`knowledge_url` is http://knowledge-service:8001 - a Kubernetes Service name
resolved by CoreDNS. Good service-to-service habits shown here:
  * explicit timeouts (a hung dependency must not hang us)
  * a small number of retries, only for idempotent reads and only on
    connection errors / 5xx (never retry a 4xx - it will fail again)
  * propagate X-Request-ID so logs of both services can be joined
  * one shared connection pool (httpx.AsyncClient) for the whole process
"""
import asyncio
import logging

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .logging_setup import log_extra, request_id_var

log = logging.getLogger(__name__)


class KnowledgeUnavailable(RuntimeError):
    pass


class KnowledgeContractError(KnowledgeUnavailable):
    """knowledge-service answered, but not in the agreed shape (e.g. after a
    deploy of an incompatible version). A subclass of KnowledgeUnavailable, so
    the agents degrade gracefully instead of crashing with a KeyError."""


# ---- the contract with knowledge-service, checked on every response ---------------
class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")   # new fields upstream are fine
    doc_id: str = Field(min_length=1)
    filename: str
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    score: float = Field(ge=-1, le=1)                         # cosine similarity
    text: str


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="allow")                  # the UI shows the extra fields
    doc_id: str = Field(min_length=1)
    filename: str
    pages: int | None = None
    total_chunks: int | None = None


class _SearchResponse(BaseModel):
    results: list[SearchHit]


# a TypeAdapter validates types that are not a BaseModel, e.g. a bare JSON list
_DOCUMENTS = TypeAdapter(list[DocumentSummary])


def _contract(exc: ValidationError, what: str) -> KnowledgeContractError:
    first = exc.errors(include_url=False)[0]
    where = ".".join(str(x) for x in first["loc"])
    return KnowledgeContractError(f"knowledge-service returned an unexpected {what} "
                                  f"({where}: {first['msg']})")


class KnowledgeClient:
    def __init__(self, base_url: str, timeout_s: float,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=3.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            transport=transport,
        )

    @staticmethod
    def _headers() -> dict:
        return {"X-Request-ID": request_id_var.get()}

    async def _read(self, method: str, path: str, retries: int = 2, **kw) -> httpx.Response:
        delay = 0.2
        for attempt in range(retries + 1):
            try:
                resp = await self._http.request(method, path, headers=self._headers(), **kw)
                if resp.status_code < 500:
                    return resp
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError,
                    httpx.PoolTimeout) as exc:
                err = f"{type(exc).__name__}: {exc}"
            log.warning("knowledge-service call failed",
                        extra=log_extra(path=path, attempt=attempt + 1, error=err))
            if attempt < retries:
                await asyncio.sleep(delay)
                delay *= 2                       # exponential backoff
        raise KnowledgeUnavailable(f"knowledge-service unavailable ({err})")

    async def search(self, query: str, top_k: int, doc_ids: list[str] | None = None,
                     score_threshold: float | None = None, embedding_model: str | None = None
                     ) -> list[SearchHit]:
        body = {"query": query, "top_k": top_k}
        if doc_ids:
            body["doc_ids"] = doc_ids
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        if embedding_model:
            body["embedding_model"] = embedding_model
        resp = await self._read("POST", "/v1/search", json=body)
        if resp.status_code >= 400:          # e.g. 402 no credits, 422 unknown embedding model
            raise KnowledgeUnavailable(f"search failed (HTTP {resp.status_code}): {resp.text[:200]}")
        try:
            return _SearchResponse.model_validate_json(resp.content).results
        except ValidationError as exc:
            raise _contract(exc, "search response") from exc

    async def list_documents(self, embedding_model: str | None = None) -> list[DocumentSummary]:
        params = {"embedding_model": embedding_model} if embedding_model else None
        resp = await self._read("GET", "/v1/documents", params=params)
        resp.raise_for_status()
        try:
            return _DOCUMENTS.validate_json(resp.content)
        except ValidationError as exc:
            raise _contract(exc, "document list") from exc

    # ---- pass-through used by the web UI (Backend-for-Frontend) ------------------
    async def options(self) -> dict:
        resp = await self._read("GET", "/v1/options", retries=0)
        resp.raise_for_status()
        return resp.json()

    async def upload(self, filename: str, data: bytes, content_type: str,
                     params: dict | None = None) -> httpx.Response:
        try:   # uploads are not retried: not idempotent from the caller's view
            return await self._http.post("/v1/documents", headers=self._headers(), params=params,
                                         files={"file": (filename, data, content_type)},
                                         timeout=httpx.Timeout(180.0, connect=3.0))
        except httpx.HTTPError as exc:
            raise KnowledgeUnavailable(str(exc)) from exc

    async def delete(self, doc_id: str, embedding_model: str | None = None) -> httpx.Response:
        params = {"embedding_model": embedding_model} if embedding_model else None
        return await self._read("DELETE", f"/v1/documents/{doc_id}", retries=1, params=params)

    async def close(self) -> None:
        await self._http.aclose()
