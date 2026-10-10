import json
import os
import sys

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Settings  # noqa: E402
from app.knowledge_client import KnowledgeClient  # noqa: E402
from app.main import create_app  # noqa: E402
from app.persistence import in_memory  # noqa: E402

PASSAGES = [
    {"doc_id": "d1", "filename": "underwriting-guidelines.md", "page": 1, "chunk_index": 2,
     "score": 0.82, "text": "Risks above USD 250 million TIV must be referred to the chief underwriter."},
    {"doc_id": "d2", "filename": "claims-sop.md", "page": 1, "chunk_index": 0,
     "score": 0.41, "text": "First notice of loss is acknowledged within 24 hours."},
]
DOC_ID = "0123456789abcdef"        # knowledge-service doc ids: sha256[:16]
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


class FakeChat(BaseChatModel):
    """A scripted chat model for ONE role. Script items:
        "text"                          -> an answer (streamed word by word)
        ("tools", [(name, args), ...])  -> tool calls (for create_agent workers)
        BaseModel / dict                -> a structured-output result
        Exception                       -> raised
    An exhausted script returns text "(script exhausted)" or, for structured
    output, None (so the node uses its fallback). Every prompt is recorded."""

    script: list = Field(default_factory=list)
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def with_structured_output(self, schema, **kw):
        """Mimics include_raw=True: the decision arrives as a raw tool call.
        A BaseModel item is dumped to its arguments; a dict item is sent as-is,
        so a test can script INVALID output and watch the validator react."""
        async def run(messages):
            self.calls.append(list(messages))
            if not self.script:
                return {"raw": AIMessage("no tool call"), "parsed": None, "parsing_error": None}
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            args = item.model_dump() if isinstance(item, BaseModel) else item
            raw = AIMessage("", tool_calls=[{"name": schema.__name__, "args": args,
                                             "id": f"call_{len(self.calls)}", "type": "tool_call"}])
            return {"raw": raw, "parsed": None, "parsing_error": None}
        return RunnableLambda(run)

    def _next(self, messages):
        self.calls.append(list(messages))
        item = self.script.pop(0) if self.script else "(script exhausted)"
        if isinstance(item, Exception):
            raise item
        return item

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        item = self._next(messages)
        if isinstance(item, tuple):
            msg = AIMessage("", tool_calls=_tool_calls(item[1]), usage_metadata=USAGE)
        else:
            msg = AIMessage(item, usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(self, messages, stop=None, run_manager=None, **kw):
        item = self._next(messages)
        if isinstance(item, tuple):
            chunks = [ToolCallChunk(name=c["name"], args=json.dumps(c["args"]), id=c["id"], index=i)
                      for i, c in enumerate(_tool_calls(item[1]))]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks,
                                                             usage_metadata=USAGE))
            return
        for word in item.split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            if run_manager:
                await run_manager.on_llm_new_token(word + " ", chunk=chunk)
            yield chunk
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=USAGE))


def _tool_calls(calls):
    return [{"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"} for i, (n, a) in enumerate(calls)]


class FakeHub:
    """Stands in for ModelHub: one FakeChat per role, scripted by the test."""

    def __init__(self, scripts: dict[str, list]):
        self.models = {role: FakeChat(script=list(items)) for role, items in scripts.items()}
        self.params: list[dict] = []

    def chat(self, cfg, role):
        self.params.append({"role": role, "model": cfg.llm_model, "temperature": cfg.temperature,
                            "max_tokens": cfg.max_tokens})
        return self.models.setdefault(role, FakeChat())

    def structured(self, cfg, role, schema):
        return self.chat(cfg, role).with_structured_output(schema)

    def prompts(self, role) -> list[list]:
        return self.models[role].calls if role in self.models else []


def prompt_text(messages) -> str:
    return "\n".join(str(x.content) for x in messages)


def knowledge_transport(fail: bool = False):
    """Fake knowledge-service at the HTTP level (httpx.MockTransport), so the
    real KnowledgeClient code - retries, headers, JSON parsing - is exercised."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if fail:
            return httpx.Response(503, json={"detail": "down"})
        path = request.url.path
        if path == "/v1/options":
            return httpx.Response(200, json={"embedding_models": ["openai/text-embedding-3-small"],
                                             "default_embedding_model": "openai/text-embedding-3-small",
                                             "chunk_size": {"default": 800, "min": 200, "max": 4000, "step": 50},
                                             "chunk_overlap": {"default": 120, "min": 0, "max": 2000, "step": 10},
                                             "top_k": {"default": 5, "min": 1, "max": 20, "step": 1},
                                             "score_threshold": {"default": 0.25, "min": 0, "max": 0.9,
                                                                 "step": 0.05}})
        if path == "/v1/search":
            body = json.loads(request.content)
            return httpx.Response(200, json={"query": body["query"],
                                             "results": PASSAGES[: body["top_k"]], "took_ms": 3})
        if path == "/v1/documents" and request.method == "GET":
            return httpx.Response(200, json=[{"doc_id": DOC_ID, "filename": "underwriting-guidelines.md",
                                              "pages": 1, "total_chunks": 3,
                                              "uploaded_at": "2026-09-01T00:00:00+00:00"}])
        if path == "/v1/documents" and request.method == "POST":
            return httpx.Response(201, json={"doc_id": "d9", "filename": "x.md", "status": "indexed",
                                             "pages": 1, "chunks": 1, "took_ms": 5})
        if path == f"/v1/documents/{DOC_ID}" and request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler), seen


def searches(seen) -> list[dict]:
    return [json.loads(r.content) for r in seen if r.url.path == "/v1/search"]


def sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture()
def settings():
    return Settings(openrouter_api_key="test-key", log_level="WARNING", rate_limit_per_minute=100,
                    agent_max_steps=4, history_max_messages=4, database_url="")


@pytest.fixture()
def make_client(settings):
    """Factory: an app with scripted models, fake Redis, fake knowledge-service
    and in-memory LangGraph persistence."""
    clients = []

    def _make(scripts=None, knowledge_fail=False, **overrides):
        s = settings.model_copy(update=overrides)
        hub = FakeHub(scripts or {})
        transport, seen = knowledge_transport(knowledge_fail)
        kc = KnowledgeClient("http://knowledge", 5, transport=transport)
        server = fakeredis.FakeServer()
        r = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
        r.fake_server = server
        app = create_app(s, models=hub, redis_client=r, knowledge=kc, persistence=in_memory())
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, hub, seen, r

    yield _make
    for c in clients:
        c.__exit__(None, None, None)

