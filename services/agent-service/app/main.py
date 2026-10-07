"""agent-service: the GenAI agent that answers questions using knowledge-service.

    GET    /                                web chat UI
    POST   /v1/chat                         one question -> JSON answer
    POST   /v1/chat/stream                  one question -> Server-Sent Events stream
    GET    /v1/sessions/{id}                conversation history
    DELETE /v1/sessions/{id}                forget a conversation
    GET    /v1/info                         model, prompt version, track
    GET    /v1/options                      allowed models + limits for the UI settings panel
    GET|POST|DELETE /api/knowledge/...      pass-through to knowledge-service (for the UI)
    GET    /healthz  /readyz  /metrics
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as m
from .agent import Agent
from .config import Settings, get_settings
from .knowledge_client import KnowledgeClient, KnowledgeUnavailable
from .llm import LLMClient, OpenRouterClient
from .logging_setup import log_extra, request_id_var, setup_logging
from .memory import ConversationMemory
from .options import OptionError, RunConfig, agent_options, resolve
from .ratelimit import RateLimiter
from .schemas import SESSION_PATTERN, ChatRequest, ChatResponse, InfoResponse
from .tracing import setup_tracing

log = logging.getLogger("agent")
STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, llm: LLMClient | None = None,
               redis_client: redis.Redis | None = None,
               knowledge: KnowledgeClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.app_name, settings.log_level)
    m.APP_INFO.info({"version": settings.app_version, "track": settings.track,
                     "model": settings.llm_model, "prompt_version": settings.prompt_version})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        r = redis_client or redis.from_url(settings.redis_url, decode_responses=True,
                                           socket_timeout=2, socket_connect_timeout=2,
                                           health_check_interval=30)
        kc = knowledge or KnowledgeClient(settings.knowledge_url, settings.knowledge_timeout_s)
        app.state.redis = r
        app.state.knowledge = kc
        app.state.limiter = RateLimiter(r, settings.rate_limit_per_minute)
        app.state.memory = ConversationMemory(r, settings.session_ttl_s, settings.history_max_messages)
        app.state.agent = Agent(llm or OpenRouterClient(settings), app.state.memory, kc, settings)
        if not settings.openrouter_api_key.get_secret_value() and llm is None:
            log.error("OPENROUTER_API_KEY is empty - /readyz will report not ready")
        log.info("ready", extra=log_extra(model=settings.llm_model, track=settings.track,
                                          knowledge=settings.knowledge_url))
        yield
        await kc.close()
        await r.aclose()
        log.info("shutdown complete")

    app = FastAPI(title="DocuMind agent-service", version=settings.app_version, lifespan=lifespan)
    setup_tracing(app, settings.app_name, settings.app_version)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
            response.headers["X-App-Track"] = settings.track
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if not route.startswith(("/metrics", "/healthz", "/readyz", "/static")):
                elapsed = time.perf_counter() - start
                m.HTTP_REQUESTS.labels(route, request.method, str(status)).inc()
                m.HTTP_LATENCY.labels(route, request.method).observe(elapsed)
                log.info("request", extra=log_extra(method=request.method, route=route,
                                                    status=status, ms=round(elapsed * 1000, 1)))
            request_id_var.reset(token)

    def client_key(request: Request) -> str:
        # behind ingress-nginx the real client is the first X-Forwarded-For entry
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    async def rate_limited(request: Request) -> None:
        allowed, remaining, reset = await request.app.state.limiter.check(client_key(request))
        if not allowed:
            m.RATE_LIMITED.inc()
            m.CHAT_REQUESTS.labels("rate_limited").inc()
            raise HTTPException(429, "Too many requests - slow down.",
                                headers={"Retry-After": str(reset)})

    # ---------------------------------------------------------------- health
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint():
        """Prometheus scrapes this every 30 s (see the ServiceMonitor)."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request):
        """Ready = "this pod can usefully answer requests", NOT "all my
        dependencies are up".
          * no API key            -> useless, report NOT ready (503)
          * Redis down            -> chat still works (memory and rate limiting
                                     fail open), so stay ready but say 'degraded'
          * knowledge-service down -> not checked at all: the agent can still say
                                     'the knowledge base is unavailable'
        If readiness depended on shared services, one Redis blip would pull
        EVERY agent pod out of the Service at once (a cascading failure)."""
        if llm is None and not settings.openrouter_api_key.get_secret_value():
            return JSONResponse({"status": "not ready", "problems": ["OPENROUTER_API_KEY not set"]},
                                status_code=503)
        degraded = []
        try:
            await request.app.state.redis.ping()
        except redis.RedisError as exc:
            degraded.append(f"redis: {exc}")
        return {"status": "ready", "degraded": degraded}

    @app.get("/v1/info", response_model=InfoResponse)
    async def info():
        return InfoResponse(service=settings.app_name, version=settings.app_version,
                            track=settings.track, model=settings.llm_model,
                            fallback_models=settings.fallback_models,
                            prompt_version=settings.prompt_version,
                            max_steps=settings.agent_max_steps)

    @app.get("/v1/options")
    async def options(request: Request):
        """Merged settings for the UI: generation/retrieval limits from this
        service, embedding models and chunking limits from knowledge-service."""
        out = {"agent": agent_options(settings)}
        try:
            out["knowledge"] = await request.app.state.knowledge.options()
        except Exception as exc:  # noqa: BLE001 - the UI still works with agent options only
            out["knowledge"] = None
            out["knowledge_error"] = str(exc)
        return out

    # ---------------------------------------------------------------- chat
    def _session(body: ChatRequest) -> str:
        return body.session_id or uuid.uuid4().hex

    def _config(body: ChatRequest) -> RunConfig:
        try:
            return resolve(body.options, settings)
        except OptionError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(rate_limited)])
    async def chat(request: Request, body: ChatRequest):
        agent: Agent = request.app.state.agent
        cfg = _config(body)
        trace: list[dict] = []
        done, error = None, None
        start = time.perf_counter()
        m.INFLIGHT.inc()
        try:
            # Always consume the generator to the END (no return/raise inside the
            # loop): the agent holds an OpenTelemetry span open across its yields,
            # and abandoning it half-way would close it later, in another context.
            async for ev in agent.run(_session(body), body.message, cfg):
                if ev.type == "step" and ev.data["status"] != "started":
                    trace.append(ev.data)
                elif ev.type == "error":
                    error = ev.data
                elif ev.type == "done":
                    done = ev.data
        finally:
            m.INFLIGHT.dec()
        if error:
            raise HTTPException(502, error["message"])
        if done is None:
            raise HTTPException(500, "agent finished without an answer")
        m.CHAT_DURATION.observe(time.perf_counter() - start)
        return ChatResponse(**done, trace=trace)

    @app.post("/v1/chat/stream", dependencies=[Depends(rate_limited)])
    async def chat_stream(request: Request, body: ChatRequest):
        agent: Agent = request.app.state.agent
        session_id = _session(body)
        cfg = _config(body)                  # validate BEFORE starting the stream

        async def events():
            start = time.perf_counter()
            m.INFLIGHT.inc()
            try:
                async for ev in agent.run(session_id, body.message, cfg):
                    yield ev.sse()
                m.CHAT_DURATION.observe(time.perf_counter() - start)
            finally:
                m.INFLIGHT.dec()

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # tell nginx/ingress-nginx not to buffer the stream
        })

    @app.get("/v1/sessions/{session_id}")
    async def get_session(request: Request, session_id: str):
        _check_session(session_id)
        return {"session_id": session_id,
                "messages": await request.app.state.memory.load(session_id)}

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(request: Request, session_id: str):
        _check_session(session_id)
        await request.app.state.memory.clear(session_id)
        return Response(status_code=204)

    # ---------------------------------------------------------------- UI + BFF proxy
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/knowledge/v1/documents")
    async def kb_list(request: Request, embedding_model: str | None = Query(None)):
        try:
            return await request.app.state.knowledge.list_documents(embedding_model)
        except KnowledgeUnavailable as exc:
            raise HTTPException(503, f"knowledge-service unavailable: {exc}") from exc

    @app.post("/api/knowledge/v1/documents")
    async def kb_upload(request: Request, file: UploadFile = File(...),
                        chunk_size: int | None = Query(None), chunk_overlap: int | None = Query(None),
                        embedding_model: str | None = Query(None)):
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
        try:
            params = {k: v for k, v in {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                                        "embedding_model": embedding_model}.items() if v is not None}
            resp = await request.app.state.knowledge.upload(
                file.filename or "upload.txt", data, file.content_type or "application/octet-stream", params)
        except KnowledgeUnavailable as exc:
            raise HTTPException(503, f"knowledge-service unavailable: {exc}") from exc
        return Response(resp.content, status_code=resp.status_code,
                        media_type=resp.headers.get("content-type"))

    @app.delete("/api/knowledge/v1/documents/{doc_id}")
    async def kb_delete(request: Request, doc_id: str, embedding_model: str | None = Query(None)):
        try:
            resp = await request.app.state.knowledge.delete(doc_id, embedding_model)
        except KnowledgeUnavailable as exc:
            raise HTTPException(503, f"knowledge-service unavailable: {exc}") from exc
        return Response(resp.content, status_code=resp.status_code)

    return app


def _check_session(session_id: str) -> None:
    import re
    if not re.fullmatch(SESSION_PATTERN, session_id):
        raise HTTPException(422, "invalid session id")


app = create_app()
