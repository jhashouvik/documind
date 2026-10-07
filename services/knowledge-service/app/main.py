"""knowledge-service: ingest documents and answer semantic-search queries.

    POST   /v1/documents             upload a PDF / TXT / MD file
                                     (?chunk_size=&chunk_overlap=&embedding_model=&replace=)
    POST   /v1/documents/text        ingest raw text (handy for scripts)
    GET    /v1/documents             list documents of one embedding model (?embedding_model=)
    GET    /v1/documents/{doc_id}    one document
    DELETE /v1/documents/{doc_id}    remove a document and all its chunks
    POST   /v1/search                semantic search (used by agent-service)
    GET    /v1/options               allowed models, defaults and limits (for the UI)
    GET    /v1/info                  collections and settings
    GET    /healthz                  liveness: the process is alive
    GET    /readyz                   readiness: Qdrant reachable and an API key configured
    GET    /metrics                  Prometheus metrics
"""
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from qdrant_client import AsyncQdrantClient

from . import metrics as m
from .chunking import chunk_pages
from .config import Settings, get_settings
from .embeddings import Embedder, EmbeddingError, build_embedder
from .logging_setup import log_extra, request_id_var, setup_logging
from .parsing import EmptyDocument, Page, UnsupportedFileType, extract_pages
from .schemas import (CollectionInfo, DocumentInfo, IngestResponse, InfoResponse, OptionsResponse,
                      Range, SearchRequest, SearchResponse, SearchResult, TextIngestRequest)
from .store import StoreRegistry, VectorStore
from .tracing import setup_tracing, span

log = logging.getLogger("knowledge")

CONTENT_TYPES = {".pdf": "application/pdf", ".md": "text/markdown",
                 ".markdown": "text/markdown", ".txt": "text/plain"}


def _content_type(filename: str) -> str:
    ext = filename.lower()[filename.rfind("."):] if "." in filename else ""
    return CONTENT_TYPES.get(ext, "application/octet-stream")


def _embedding_input(filename: str, text: str) -> str:
    """Prefix each chunk with its document title before embedding
    ("contextual chunk headers"). A chunk that says 'the limit is 50M' is far
    easier to retrieve when the vector also knows it came from
    'property underwriting guidelines'. We store the raw text, not this."""
    title = filename.rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
    return f"Document: {title}\n\n{text}"


def create_app(settings: Settings | None = None, embedder: Embedder | None = None,
               qdrant: AsyncQdrantClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.app_name, settings.log_level)
    m.APP_INFO.info({"version": settings.app_version, "embedding_provider": settings.embedding_provider,
                     "default_embedding_model": settings.embedding_model})

    # ---------------------------------------------------------------- validation helpers
    def pick_model(model: str | None) -> str:
        model = model or settings.embedding_model
        if model not in settings.allowed_embedding_models:
            raise HTTPException(422, f"embedding_model must be one of {settings.allowed_embedding_models}")
        return model

    def pick_chunking(size: int | None, overlap: int | None) -> tuple[int, int]:
        size = size or settings.chunk_size
        overlap = settings.chunk_overlap if overlap is None else overlap
        if not settings.min_chunk_size <= size <= settings.max_chunk_size:
            raise HTTPException(422, f"chunk_size must be {settings.min_chunk_size}-{settings.max_chunk_size}")
        if not 0 <= overlap <= size // 2:
            raise HTTPException(422, "chunk_overlap must be between 0 and half the chunk size")
        return size, overlap

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ---- startup: runs once per pod, before any request is served ----------
        m.READY.set(0)
        emb = embedder or build_embedder(settings)
        client = qdrant or AsyncQdrantClient(url=settings.qdrant_url,
                                             api_key=settings.qdrant_api_key, timeout=30)
        registry = StoreRegistry(client, settings.collection_prefix, emb)
        probe = VectorStore(client, "-", 0)
        await probe.wait_until_available(settings.qdrant_connect_retries, settings.qdrant_retry_delay_s)
        app.state.embedder, app.state.registry, app.state.ready = emb, registry, True
        m.READY.set(1)
        log.info("ready", extra=log_extra(qdrant=settings.qdrant_url, provider=settings.embedding_provider,
                                          default_model=settings.embedding_model))
        yield
        # ---- shutdown: Kubernetes sent SIGTERM ---------------------------------------
        app.state.ready = False
        m.READY.set(0)
        await emb.close()
        await client.close()
        log.info("shutdown complete")

    app = FastAPI(title="DocuMind knowledge-service", version=settings.app_version, lifespan=lifespan)
    app.state.ready = False
    setup_tracing(app, settings.app_name, settings.app_version)

    # ---------------------------------------------------------------- middleware
    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers["X-App-Version"] = settings.app_version
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            if not path.startswith(("/metrics", "/healthz", "/readyz")):
                elapsed = time.perf_counter() - start
                m.HTTP_REQUESTS.labels(path, request.method, str(status)).inc()
                m.HTTP_LATENCY.labels(path, request.method).observe(elapsed)
                log.info("request", extra=log_extra(method=request.method, route=path, status=status,
                                                    ms=round(elapsed * 1000, 1)))
            request_id_var.reset(token)

    @app.exception_handler(EmbeddingError)
    async def embedding_error(_: Request, exc: EmbeddingError):
        m.EMBED_ERRORS.labels(str(exc.status)).inc()
        log.error("embedding failed", extra=log_extra(error=str(exc), status=exc.status))
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    # ---------------------------------------------------------------- health + meta
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        """Prometheus scrapes this every 30 s (see the ServiceMonitor)."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz():
        """Ready = Qdrant reachable and an API key configured. OpenRouter itself
        is NOT checked: an internet blip must not pull every pod out of service
        (and we would pay for a probe call every 10 s)."""
        registry: StoreRegistry | None = getattr(app.state, "registry", None)
        problems = []
        if not app.state.ready or registry is None or not await registry.ping():
            problems.append("qdrant not reachable")
        if settings.embedding_provider == "openrouter" and embedder is None \
                and not settings.openrouter_api_key.get_secret_value():
            problems.append("OPENROUTER_API_KEY not set")
        if problems:
            return JSONResponse({"status": "not ready", "problems": problems}, status_code=503)
        return {"status": "ready"}

    @app.get("/v1/options", response_model=OptionsResponse)
    async def options():
        return OptionsResponse(
            embedding_models=settings.allowed_embedding_models,
            default_embedding_model=settings.embedding_model,
            chunk_size=Range(default=settings.chunk_size, min=settings.min_chunk_size,
                             max=settings.max_chunk_size, step=50),
            chunk_overlap=Range(default=settings.chunk_overlap, min=0, max=settings.max_chunk_size // 2,
                                step=10),
            top_k=Range(default=settings.default_top_k, min=1, max=settings.max_top_k, step=1),
            score_threshold=Range(default=settings.score_threshold, min=0, max=0.9, step=0.05))

    @app.get("/v1/info", response_model=InfoResponse)
    async def info(request: Request):
        registry: StoreRegistry = request.app.state.registry
        cols = []
        for name in await registry.existing_collections():
            cols.append(CollectionInfo(collection=name,
                                       points=(await registry.client.count(name, exact=True)).count))
        return InfoResponse(service=settings.app_name, version=settings.app_version,
                            embedding_provider=settings.embedding_provider,
                            default_embedding_model=settings.embedding_model, collections=cols,
                            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    # ---------------------------------------------------------------- ingestion
    async def _ingest(request: Request, filename: str, data: bytes, pages: list[Page], replace: bool,
                      model: str, size: int, overlap: int) -> IngestResponse:
        start = time.perf_counter()
        store = await request.app.state.registry.get(model)
        emb: Embedder = request.app.state.embedder

        # content-addressed id: the same bytes always get the same doc_id
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        existing = await store.get_document(doc_id)
        same_settings = existing and existing.get("chunk_size") == size and existing.get("chunk_overlap") == overlap
        if existing and same_settings and not replace:
            m.INGEST_DOCUMENTS.labels("duplicate").inc()
            return IngestResponse(doc_id=doc_id, filename=existing["filename"], status="duplicate",
                                  pages=existing.get("pages") or len(pages),
                                  chunks=existing.get("total_chunks") or 0, chunk_size=size,
                                  chunk_overlap=overlap, embedding_model=model,
                                  took_ms=int((time.perf_counter() - start) * 1000))
        if existing:                    # replace requested, or chunking changed -> rebuild
            await store.delete_document(doc_id)

        with span("chunk document", **{"documind.chunk_size": size, "documind.chunk_overlap": overlap}) as s:
            chunks = chunk_pages(pages, size, overlap)
            s.set_attribute("documind.chunks", len(chunks))
        vectors = await emb.embed([_embedding_input(filename, c.text) for c in chunks], model, "documents")

        meta = {"doc_id": doc_id, "filename": filename, "content_type": _content_type(filename),
                "pages": len(pages), "size_bytes": len(data), "embedding_model": model,
                "chunk_size": size, "chunk_overlap": overlap,
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with span("qdrant upsert", **{"db.system": "qdrant", "db.collection.name": store.collection}):
            await store.upsert_chunks(meta, chunks, vectors)

        status = "reindexed" if existing else "indexed"
        m.INGEST_DOCUMENTS.labels(status).inc()
        m.INGEST_CHUNKS.inc(len(chunks))
        m.CHUNKS_PER_DOC.observe(len(chunks))
        took = int((time.perf_counter() - start) * 1000)
        log.info("document indexed", extra=log_extra(doc_id=doc_id, filename=filename, pages=len(pages),
                                                     chunks=len(chunks), model=model, chunk_size=size,
                                                     chunk_overlap=overlap, ms=took))
        return IngestResponse(doc_id=doc_id, filename=filename, status=status, pages=len(pages),
                              chunks=len(chunks), chunk_size=size, chunk_overlap=overlap,
                              embedding_model=model, took_ms=took)

    @app.post("/v1/documents", response_model=IngestResponse, status_code=201)
    async def upload_document(request: Request, response: Response, file: UploadFile = File(...),
                              replace: bool = Query(False, description="Re-index if it exists"),
                              chunk_size: int | None = Query(None), chunk_overlap: int | None = Query(None),
                              embedding_model: str | None = Query(None)):
        model = pick_model(embedding_model)
        size, overlap = pick_chunking(chunk_size, chunk_overlap)
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)          # never read more than the limit
        if len(data) > max_bytes:
            m.INGEST_DOCUMENTS.labels("rejected").inc()
            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
        filename = (file.filename or "upload.txt").split("/")[-1].split("\\")[-1]
        try:
            with span("parse document", **{"documind.filename": filename}):
                pages = await run_in_threadpool(extract_pages, filename, data)
        except UnsupportedFileType as exc:
            m.INGEST_DOCUMENTS.labels("rejected").inc()
            raise HTTPException(415, str(exc)) from exc
        except EmptyDocument as exc:
            m.INGEST_DOCUMENTS.labels("rejected").inc()
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - corrupt PDFs raise many things
            m.INGEST_DOCUMENTS.labels("failed").inc()
            raise HTTPException(422, f"Could not parse file: {exc}") from exc
        result = await _ingest(request, filename, data, pages, replace, model, size, overlap)
        if result.status == "duplicate":
            response.status_code = 200
        return result

    @app.post("/v1/documents/text", response_model=IngestResponse, status_code=201)
    async def ingest_text(request: Request, response: Response, body: TextIngestRequest,
                          replace: bool = False):
        model = pick_model(body.embedding_model)
        size, overlap = pick_chunking(body.chunk_size, body.chunk_overlap)
        title = body.title if "." in body.title else f"{body.title}.md"
        data = body.text.encode()
        try:
            pages = extract_pages(title, data)
        except (UnsupportedFileType, EmptyDocument) as exc:
            raise HTTPException(422, str(exc)) from exc
        result = await _ingest(request, title, data, pages, replace, model, size, overlap)
        if result.status == "duplicate":
            response.status_code = 200
        return result

    @app.get("/v1/documents", response_model=list[DocumentInfo])
    async def list_documents(request: Request, embedding_model: str | None = Query(None)):
        store = await request.app.state.registry.get(pick_model(embedding_model))
        return await store.list_documents()

    @app.get("/v1/documents/{doc_id}", response_model=DocumentInfo)
    async def get_document(request: Request, doc_id: str, embedding_model: str | None = Query(None)):
        store = await request.app.state.registry.get(pick_model(embedding_model))
        doc = await store.get_document(doc_id)
        if not doc:
            raise HTTPException(404, "document not found")
        return doc

    @app.delete("/v1/documents/{doc_id}", status_code=204)
    async def delete_document(request: Request, doc_id: str, embedding_model: str | None = Query(None)):
        store = await request.app.state.registry.get(pick_model(embedding_model))
        deleted = await store.delete_document(doc_id)
        if not deleted:
            raise HTTPException(404, "document not found")
        log.info("document deleted", extra=log_extra(doc_id=doc_id, chunks=deleted))
        return Response(status_code=204)

    # ---------------------------------------------------------------- search
    @app.post("/v1/search", response_model=SearchResponse)
    async def search(request: Request, body: SearchRequest):
        start = time.perf_counter()
        model = pick_model(body.embedding_model)
        store: VectorStore = await request.app.state.registry.get(model)
        emb: Embedder = request.app.state.embedder
        top_k = min(body.top_k or settings.default_top_k, settings.max_top_k)
        threshold = settings.score_threshold if body.score_threshold is None else body.score_threshold

        vector = (await emb.embed([body.query], model, "query"))[0]
        with span("qdrant search", **{"db.system": "qdrant", "db.collection.name": store.collection,
                                      "documind.top_k": top_k, "documind.score_threshold": threshold}) as s:
            hits = await store.search(vector, top_k, threshold, body.doc_ids)
            s.set_attribute("documind.results", len(hits))
        elapsed = time.perf_counter() - start
        m.SEARCH_LATENCY.observe(elapsed)
        m.SEARCH_RESULTS.observe(len(hits))
        if hits:
            m.SEARCH_TOP_SCORE.observe(hits[0].score)
        return SearchResponse(query=body.query, embedding_model=model,
                              results=[SearchResult(**h.__dict__) for h in hits],
                              took_ms=int(elapsed * 1000))

    return app


app = create_app()
