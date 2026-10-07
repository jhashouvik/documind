"""Tests the real OpenRouterClient against a fake OpenRouter speaking the
OpenAI streaming wire format, including tool-call arguments split across
several chunks - the trickiest part of streaming agents."""
import json

import httpx
import pytest

from app.config import Settings
from app.llm import LLMError, OpenRouterClient


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _chunk(delta=None, finish=None, usage=None):
    c = {"id": "gen-1", "object": "chat.completion.chunk", "created": 1, "model": "openai/gpt-4o-mini",
         "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def _client(handler, **kw):
    s = Settings(openrouter_api_key="k", llm_max_retries=0, **kw)
    return OpenRouterClient(s, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _collect(client, tools=None):
    return [ev async for ev in client.stream_chat([{"role": "user", "content": "hi"}], tools)]


@pytest.mark.anyio
async def test_text_stream():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        seen["headers"] = req.headers
        return _sse([_chunk({"role": "assistant", "content": "Hel"}), _chunk({"content": "lo"}),
                     _chunk({}, "stop"), _chunk(usage={"prompt_tokens": 5, "completion_tokens": 2,
                                                       "total_tokens": 7})])
    events = await _collect(_client(handler, llm_fallback_models="meta-llama/llama-3.3-70b-instruct"))
    assert "".join(e.text for e in events if e.kind == "content") == "Hello"
    end = events[-1]
    assert end.kind == "end" and end.tool_calls == [] and end.usage["prompt_tokens"] == 5
    assert seen["body"]["stream"] is True
    assert seen["body"]["models"] == ["openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct"]
    assert seen["headers"]["authorization"] == "Bearer k"
    assert seen["headers"]["x-title"] == "DocuMind"


@pytest.mark.anyio
async def test_tool_call_fragments_are_reassembled():
    def handler(req):
        return _sse([
            _chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                                                         "function": {"name": "search_knowledge_base", "arguments": ""}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"query": "TI'}}]}),
            _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'V limit"}'}}]}),
            _chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                                    "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'}}]}),
            _chunk({}, "tool_calls"),
        ])
    events = await _collect(_client(handler), tools=[{"type": "function", "function": {"name": "x"}}])
    calls = events[-1].tool_calls
    assert [(c.id, c.name) for c in calls] == [("call_a", "search_knowledge_base"), ("call_b", "calculator")]
    assert json.loads(calls[0].arguments) == {"query": "TIV limit"}
    assert events[-1].finish_reason == "tool_calls"


@pytest.mark.anyio
@pytest.mark.parametrize("status,needle", [(401, "API key"), (402, "credits"), (404, "tool calling"),
                                           (429, "rate limit")])
async def test_errors_are_translated(status, needle):
    def handler(req):
        return httpx.Response(status, json={"error": {"message": "nope", "code": status}})
    with pytest.raises(LLMError) as err:
        await _collect(_client(handler))
    assert needle in str(err.value) and err.value.status == status
