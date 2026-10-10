"""agent-service: the DocuMind multi-agent system (LangGraph) behind a FastAPI API.

    GET    /                                  web chat UI
    POST   /v1/chat                           one question -> JSON (answer, or "interrupted")
    POST   /v1/chat/stream                    one question -> Server-Sent Events stream
    POST   /v1/chat/resume[/stream]           approve / reject a paused action (human-in-the-loop)
    GET    /v1/sessions/{id}                  conversation history (from the checkpointer)
    DELETE /v1/sessions/{id}                  forget a conversation
    GET    /v1/sessions/{id}/checkpoints      every saved state of the thread (time travel)
    POST   /v1/sessions/{id}/replay/stream    continue from an old checkpoint (fork)
    GET    /v1/graph                          the agent graph as Mermaid
    GET    /v1/info                           model, prompt version, track, persistence
    GET    /v1/options                        allowed models + limits for the UI settings panel
    GET|POST|DELETE /api/knowledge/...        pass-through to knowledge-service (for the UI)
    GET    /healthz  /readyz  /metrics
"""
import logging
import re
import socket
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import redis.asyncio as redis
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from . import metrics as m
from .config import Settings, get_settings
from .graph.builder import AGENTS, compile_graph
from .graph.runner import AgentRunner, NothingToResume
from .knowledge_client import KnowledgeClient, KnowledgeUnavailable
from .logging_setup import log_extra, request_id_var, setup_logging
from .models import ModelHub
from .options import OptionError, RunConfig, agent_options, resolve
from .persistence import Persistence, open_persistence
from .ratelimit import RateLimiter
from .events import EVENT_ADAPTER, DoneEvent, ErrorEvent, InterruptEvent, StepEvent
from .graph.tools import TOOLS
from .schemas import (SESSION_PATTERN, ChatRequest, ChatResponse, EditDecision, InfoResponse,
                      ReplayRequest, ResumeDecision)
from .tracing import setup_tracing

log = logging.getLogger("agent")
STATIC_DIR = Path(__file__).parent / "static"
HOSTNAME = socket.gethostname()
SSE_HEADERS = {"Cache-Control": "no-cache",
               "X-Accel-Buffering": "no"}    # tell nginx/ingress-nginx not to buffer the stream


def create_app(settings: Settings | None = None, models=None, redis_client: redis.Redis | None = None,
               knowledge: KnowledgeClient | None = None, persistence: Persistence | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.app_name, settings.log_level)
    m.APP_INFO.info({"version": settings.app_version, "track": settings.track, "engine": "langgraph",
                     "model": settings.llm_model, "prompt_version": settings.prompt_version})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            p = persistence or await stack.enter_async_context(open_persistence(settings))
            r = redis_client or redis.from_url(settings.redis_url, decode_responses=True,
                                               socket_timeout=2, socket_connect_timeout=2,
                                               health_check_interval=30)
            kc = knowledge or KnowledgeClient(settings.knowledge_url, settings.knowledge_timeout_s)
            graph = compile_graph(checkpointer=p.checkpointer, store=p.store)
            app.state.persistence, app.state.redis, app.state.knowledge = p, r, kc
            app.state.graph = graph
            app.state.limiter = RateLimiter(r, settings.rate_limit_per_minute)
            app.state.runner = AgentRunner(graph, settings, models or ModelHub(settings), kc)
            if not settings.openrouter_api_key.get_secret_value() and models is None:
                log.error("OPENROUTER_API_KEY is empty - /readyz will report not ready")
            log.info("ready", extra=log_extra(model=settings.llm_model, track=settings.track,
                                              persistence=p.kind, knowledge=settings.knowledge_url))
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
            # in Kubernetes the hostname is the pod name: shows which replica answered
            # (scripts/stateless-demo.sh uses it to prove any pod can continue a chat)
            response.headers["X-Served-By"] = HOSTNAME
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
          * no API key             -> useless, report NOT ready (503)
          * Redis / Postgres down  -> stay ready but say 'degraded' (rate limiting
                                      fails open; chat calls report the error)
          * knowledge-service down -> not checked: the agents can still say
                                      'the knowledge base is unavailable'
        If readiness depended on shared services, one Postgres blip would pull
        EVERY agent pod out of the Service at once (a cascading failure)."""
        if models is None and not settings.openrouter_api_key.get_secret_value():
            return JSONResponse({"status": "not ready", "problems": ["OPENROUTER_API_KEY not set"]},
                                status_code=503)
        degraded = []
        try:
            await request.app.state.redis.ping()
        except redis.RedisError as exc:
            degraded.append(f"redis: {exc}")
        if not await request.app.state.persistence.ping():
            degraded.append("postgres: not reachable")
        return {"status": "ready", "degraded": degraded}

    @app.get("/v1/info", response_model=InfoResponse)
    async def info(request: Request):
        return InfoResponse(service=settings.app_name, version=settings.app_version,
                            track=settings.track, model=settings.llm_model,
                            fallback_models=settings.fallback_models,
                            prompt_version=settings.prompt_version, max_steps=settings.agent_max_steps,
                            engine="langgraph", persistence=request.app.state.persistence.kind,
                            agents=AGENTS)

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

    @app.get("/v1/graph")
    async def graph_view(request: Request):
        """The compiled graph, sub-agents expanded (xray), as Mermaid text."""
        return {"mermaid": request.app.state.graph.get_graph(xray=1).draw_mermaid()}

    @app.get("/v1/events/schema")
    async def events_schema():
        """JSON Schema of every SSE event (a discriminated union on "type")."""
        return EVENT_ADAPTER.json_schema()

    # ---------------------------------------------------------------- chat
    def _config(body: ChatRequest) -> RunConfig:
        try:
            return resolve(body.options, settings)
        except OptionError as exc:
            raise HTTPException(422, str(exc)) from exc

    def _stream(events) -> StreamingResponse:
        async def gen():
            start = time.perf_counter()
            m.INFLIGHT.inc()
            try:
                async for ev in events:
                    yield ev.sse()
                m.CHAT_DURATION.observe(time.perf_counter() - start)
            finally:
                m.INFLIGHT.dec()
        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    async def _collect(session_id: str, events) -> ChatResponse:
        trace: list[dict] = []
        done = interrupt = error = None
        start = time.perf_counter()
        m.INFLIGHT.inc()
        try:
            # Always consume the generator to the END: the runner holds an
            # OpenTelemetry span open across its yields.
            async for ev in events:
                match ev:                     # the event classes make this an exhaustive switch
                    case StepEvent():
                        trace.append(ev.model_dump(exclude={"type"}))
                    case ErrorEvent():
                        error = ev
                    case InterruptEvent():
                        interrupt = ev
                    case DoneEvent():
                        done = ev
        finally:
            m.INFLIGHT.dec()
        if error:
            raise HTTPException(502, error.message)
        if interrupt:
            return ChatResponse(status="interrupted", session_id=session_id, trace=trace,
                                interrupt=interrupt.model_dump(exclude={"type"}),
                                version=settings.app_version, track=settings.track)
        if done is None:
            raise HTTPException(500, "agent finished without an answer")
        m.CHAT_DURATION.observe(time.perf_counter() - start)
        return ChatResponse(**done.model_dump(exclude={"type"}), trace=trace)

    def _start(request: Request, body: ChatRequest) -> tuple[str, object]:
        session_id = body.session_id or uuid.uuid4().hex
        cfg = _config(body)                  # validate BEFORE starting a stream
        return session_id, request.app.state.runner.run(session_id, body.message, cfg, body.user_id)

    async def _resume(request: Request, body: ResumeDecision):
        runner: AgentRunner = request.app.state.runner
        try:
            snap = await runner.pending(body.session_id)
        except NothingToResume as exc:
            raise HTTPException(409, str(exc)) from exc
        allowed = runner.allowed_decisions(snap)
        if allowed and body.decision not in allowed:
            raise HTTPException(422, f"'{body.decision}' is not allowed here; allowed: {sorted(allowed)}")
        if isinstance(body, EditDecision):
            actions = runner.pending_actions(snap)
            if len(actions) != 1 or actions[0].get("name") not in TOOLS:
                raise HTTPException(422, "edit needs exactly one pending tool call")
            tool = TOOLS[actions[0]["name"]]
            try:   # the human's arguments must satisfy the same schema the LLM's had to
                tool.tool_call_schema.model_validate(body.args)
            except ValidationError as exc:
                raise HTTPException(422, {"message": f"invalid arguments for {tool.name}",
                                          "errors": exc.errors(include_url=False, include_context=False)}
                                    ) from exc
        return runner.resume(snap, body.session_id, body)

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(rate_limited)])
    async def chat(request: Request, body: ChatRequest):
        session_id, events = _start(request, body)
        return await _collect(session_id, events)

    @app.post("/v1/chat/stream", dependencies=[Depends(rate_limited)])
    async def chat_stream(request: Request, body: ChatRequest):
        _, events = _start(request, body)
        return _stream(events)

    @app.post("/v1/chat/resume", response_model=ChatResponse, dependencies=[Depends(rate_limited)])
    async def chat_resume(request: Request, body: ResumeDecision):
        return await _collect(body.session_id, await _resume(request, body))

    @app.post("/v1/chat/resume/stream", dependencies=[Depends(rate_limited)])
    async def chat_resume_stream(request: Request, body: ResumeDecision):
        return _stream(await _resume(request, body))

    # ---------------------------------------------------------------- sessions (checkpointer)
    @app.get("/v1/sessions/{session_id}")
    async def get_session(request: Request, session_id: str):
        _check_session(session_id)
        return {"session_id": session_id,
                "messages": await request.app.state.runner.messages(session_id)}

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    async def delete_session(request: Request, session_id: str):
        _check_session(session_id)
        await request.app.state.runner.forget(session_id)
        return Response(status_code=204)

    @app.get("/v1/sessions/{session_id}/checkpoints")
    async def list_checkpoints(request: Request, session_id: str, limit: int = Query(50, ge=1, le=200)):
        _check_session(session_id)
        return {"session_id": session_id,
                "checkpoints": await request.app.state.runner.checkpoints(session_id, limit)}

    @app.post("/v1/sessions/{session_id}/replay/stream", dependencies=[Depends(rate_limited)])
    async def replay(request: Request, session_id: str, body: ReplayRequest):
        _check_session(session_id)
        try:
            events = await request.app.state.runner.replay(session_id, body.checkpoint_id)
        except NothingToResume as exc:
            raise HTTPException(404, str(exc)) from exc
        return _stream(events)

    # ---------------------------------------------------------------- UI + BFF proxy
    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/knowledge/v1/documents")
    async def kb_list(request: Request, embedding_model: str | None = Query(None)):
        try:
            docs = await request.app.state.knowledge.list_documents(embedding_model)
            return [d.model_dump() for d in docs]          # extra="allow": all fields for the UI
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
    if not re.fullmatch(SESSION_PATTERN, session_id):
        raise HTTPException(422, "invalid session id")


app = create_app()
