"""Qdrant vector store access.

Data model: one Qdrant *point* per chunk.
  id       deterministic UUID5 of "<doc_id>:<chunk_index>"  -> re-ingesting the
           same document overwrites instead of duplicating (idempotent writes)
  vector   384 floats from the embedding model
  payload  doc_id, filename, page, chunk_index, total_chunks, text, ...

Payload indexes on doc_id and chunk_index make "delete this document" and
"list documents" (= chunks with chunk_index 0) fast filtered operations.
"""
import asyncio
import logging
import re
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from .chunking import Chunk
from .logging_setup import log_extra

log = logging.getLogger(__name__)
POINT_NAMESPACE = uuid.UUID("6f1c3a0e-8a53-4d6b-9c7e-3b1f0c2d5e11")


class CollectionMismatch(RuntimeError):
    """The existing collection was built with a different vector size."""


@dataclass
class SearchHit:
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    score: float
    text: str


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


class VectorStore:
    def __init__(self, client: AsyncQdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    # ---------- lifecycle -------------------------------------------------------
    async def wait_until_available(self, retries: int, delay_s: float) -> None:
        """In Kubernetes there is no start-up order between pods, so we retry
        instead of crashing if Qdrant is not up yet."""
        for attempt in range(1, retries + 1):
            try:
                await self.client.get_collections()
                return
            except Exception as exc:  # noqa: BLE001 - any connection problem
                log.warning("qdrant not reachable yet",
                            extra=log_extra(attempt=attempt, error=str(exc)))
                await asyncio.sleep(delay_s)
        raise RuntimeError(f"Qdrant not reachable after {retries} attempts")

    async def ensure_collection(self) -> None:
        if await self.client.collection_exists(self.collection):
            info = await self.client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore[union-attr]
            if size != self.dim:
                raise CollectionMismatch(
                    f"collection '{self.collection}' stores {size}-d vectors but the "
                    f"embedding model produces {self.dim}-d vectors. Use a new COLLECTION "
                    f"name (and re-ingest) when you change embedding model."
                )
            return
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        await self.client.create_payload_index(
            self.collection, "doc_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
        await self.client.create_payload_index(
            self.collection, "chunk_index", field_schema=models.PayloadSchemaType.INTEGER
        )
        log.info("created collection", extra=log_extra(collection=self.collection, dim=self.dim))

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---------- writes ------------------------------------------------------------
    async def upsert_chunks(self, doc_meta: dict, chunks: list[Chunk],
                            vectors: list[list[float]], batch_size: int = 128) -> None:
        points = [
            models.PointStruct(
                id=point_id(doc_meta["doc_id"], c.index),
                vector=vec,
                payload={**doc_meta, "page": c.page, "chunk_index": c.index,
                         "total_chunks": len(chunks), "text": c.text},
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), batch_size):          # bounded request size
            await self.client.upsert(self.collection, points=points[i:i + batch_size], wait=True)

    async def delete_document(self, doc_id: str) -> int:
        flt = models.Filter(must=[models.FieldCondition(
            key="doc_id", match=models.MatchValue(value=doc_id))])
        n = (await self.client.count(self.collection, count_filter=flt, exact=True)).count
        if n:
            await self.client.delete(self.collection,
                                     points_selector=models.FilterSelector(filter=flt), wait=True)
        return n

    # ---------- reads ---------------------------------------------------------------
    async def get_document(self, doc_id: str) -> dict | None:
        records, _ = await self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0)),
            ]),
            limit=1, with_payload=True,
        )
        return self._doc_summary(records[0].payload) if records else None

    async def list_documents(self, limit: int = 200) -> list[dict]:
        """Every document has exactly one chunk with chunk_index == 0."""
        records, _ = await self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="chunk_index", match=models.MatchValue(value=0))]),
            limit=limit, with_payload=True,
        )
        docs = [self._doc_summary(r.payload) for r in records]
        return sorted(docs, key=lambda d: d["uploaded_at"], reverse=True)

    async def count_points(self) -> int:
        return (await self.client.count(self.collection, exact=True)).count

    async def search(self, vector: list[float], top_k: int, score_threshold: float | None,
                     doc_ids: list[str] | None = None) -> list[SearchHit]:
        flt = None
        if doc_ids:
            flt = models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchAny(any=doc_ids))])
        res = await self.client.query_points(
            self.collection, query=vector, limit=top_k, query_filter=flt,
            score_threshold=score_threshold, with_payload=True,
        )
        return [
            SearchHit(doc_id=p.payload["doc_id"], filename=p.payload["filename"],
                      page=p.payload["page"], chunk_index=p.payload["chunk_index"],
                      score=round(float(p.score), 4), text=p.payload["text"])
            for p in res.points
        ]

    @staticmethod
    def _doc_summary(payload: dict) -> dict:
        return {k: payload.get(k) for k in
                ("doc_id", "filename", "content_type", "pages", "total_chunks",
                 "size_bytes", "uploaded_at", "embedding_model", "chunk_size", "chunk_overlap")}


def collection_name(prefix: str, model: str) -> str:
    """openai/text-embedding-3-small -> documind__openai_text_embedding_3_small"""
    return f"{prefix}__{re.sub(r'[^a-z0-9]+', '_', model.lower()).strip('_')}"


class StoreRegistry:
    """One VectorStore (= one Qdrant collection) per embedding model, created on
    first use. The vector size is not configured anywhere: we embed a probe
    string once and use the length of the returned vector."""

    def __init__(self, client: AsyncQdrantClient, prefix: str, embedder) -> None:
        self.client = client
        self.prefix = prefix
        self.embedder = embedder
        self._stores: dict[str, VectorStore] = {}
        self._lock = asyncio.Lock()

    async def get(self, model: str) -> VectorStore:
        if model in self._stores:
            return self._stores[model]
        async with self._lock:                      # two requests must not create it twice
            if model not in self._stores:
                name = collection_name(self.prefix, model)
                if await self.client.collection_exists(name):
                    info = await self.client.get_collection(name)
                    dim = info.config.params.vectors.size  # type: ignore[union-attr]
                else:
                    dim = len((await self.embedder.embed(["dimension probe"], model, "probe"))[0])
                store = VectorStore(self.client, name, dim)
                await store.ensure_collection()
                self._stores[model] = store
        return self._stores[model]

    async def existing_collections(self) -> list[str]:
        cols = (await self.client.get_collections()).collections
        return sorted(c.name for c in cols if c.name.startswith(f"{self.prefix}__"))

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False
