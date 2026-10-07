import json
import os
import sys

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Settings  # noqa: E402
from app.knowledge_client import KnowledgeClient  # noqa: E402
from app.llm import LLMEvent, ToolCall  # noqa: E402
from app.main import create_app  # noqa: E402

PASSAGES = [
    {"doc_id": "d1", "filename": "underwriting-guidelines.md", "page": 1, "chunk_index": 2,
     "score": 0.82, "text": "Risks above USD 250 million TIV must be referred to the chief underwriter."},
    {"doc_id": "d2", "filename": "claims-sop.md", "page": 1, "chunk_index": 0,
     "score": 0.41, "text": "First notice of loss is acknowledged within 24 hours."},
]


class FakeLLM:
    """Replays a script of turns. Each turn is either
    ("text", "answer ...") or ("tools", [(name, args_dict_or_raw_string), ...]).
    Records every `messages` list it receives so tests can inspect the prompt."""

    model = "fake/model"

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.tools_offered: list[bool] = []
        self.params: list[dict] = []

    async def stream_chat(self, messages, tools, model=None, temperature=None, max_tokens=None):
        self.calls.append([dict(x) for x in messages])
        self.tools_offered.append(bool(tools))
        self.params.append({"model": model, "temperature": temperature, "max_tokens": max_tokens})
        kind, payload = self.script.pop(0) if self.script else ("text", "(script exhausted)")
        usage = {"prompt_tokens": 100, "completion_tokens": 20}
        if kind == "text":
            for word in payload.split(" "):
                yield LLMEvent("content", text=word + " ")
            yield LLMEvent("end", usage=usage, model=self.model, finish_reason="stop")
        elif kind == "error":
            raise payload
        else:
            calls = [ToolCall(id=f"call_{i}", name=n,
                              arguments=a if isinstance(a, str) else json.dumps(a))
                     for i, (n, a) in enumerate(payload)]
            yield LLMEvent("end", tool_calls=calls, usage=usage, model=self.model,
                           finish_reason="tool_calls")


def knowledge_transport(fail: bool = False):
    """Fake knowledge-service at the HTTP level (httpx.MockTransport), so the
    real KnowledgeClient code - retries, headers, JSON parsing - is exercised."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if fail:
            return httpx.Response(503, json={"detail": "down"})
        if request.url.path == "/v1/options":
            return httpx.Response(200, json={"embedding_models": ["openai/text-embedding-3-small"],
                                             "default_embedding_model": "openai/text-embedding-3-small",
                                             "chunk_size": {"default": 800, "min": 200, "max": 4000, "step": 50},
                                             "chunk_overlap": {"default": 120, "min": 0, "max": 2000, "step": 10},
                                             "top_k": {"default": 5, "min": 1, "max": 20, "step": 1},
                                             "score_threshold": {"default": 0.25, "min": 0, "max": 0.9,
                                                                 "step": 0.05}})
        if request.url.path == "/v1/search":
            body = json.loads(request.content)
            return httpx.Response(200, json={"query": body["query"],
                                             "results": PASSAGES[: body["top_k"]], "took_ms": 3})
        if request.url.path == "/v1/documents" and request.method == "GET":
            return httpx.Response(200, json=[{"doc_id": "d1", "filename": "underwriting-guidelines.md",
                                              "pages": 1, "total_chunks": 3,
                                              "uploaded_at": "2026-09-01T00:00:00+00:00"}])
        if request.url.path == "/v1/documents" and request.method == "POST":
            return httpx.Response(201, json={"doc_id": "d9", "filename": "x.md", "status": "indexed",
                                             "pages": 1, "chunks": 1, "took_ms": 5})
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler), seen


@pytest.fixture()
def settings():
    return Settings(openrouter_api_key="test-key", log_level="WARNING", rate_limit_per_minute=100,
                    agent_max_steps=4, history_max_messages=4)


@pytest.fixture()
def make_client(settings):
    """Factory: build an app with a scripted LLM, fake Redis and fake knowledge-service."""
    def _make(script, knowledge_fail=False, **overrides):
        s = settings.model_copy(update=overrides)
        llm = FakeLLM(script)
        transport, seen = knowledge_transport(knowledge_fail)
        kc = KnowledgeClient("http://knowledge", 5, transport=transport)
        server = fakeredis.FakeServer()
        r = fakeredis.FakeAsyncRedis(server=server, decode_responses=True)
        r.fake_server = server
        app = create_app(s, llm=llm, redis_client=r, knowledge=kc)
        client = TestClient(app)
        client.__enter__()
        return client, llm, seen, r
    return _make
