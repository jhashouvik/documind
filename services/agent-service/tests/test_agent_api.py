import json

from app.llm import LLMError

SEARCH = ("tools", [("search_knowledge_base", {"query": "TIV referral threshold"})])


def _sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health_ready_and_info(make_client):
    c, *_ = make_client([])
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/readyz").status_code == 200
    info = c.get("/v1/info").json()
    assert info["model"] == "openai/gpt-4o-mini" and info["track"] == "stable"


def test_ui_is_served(make_client):
    c, *_ = make_client([])
    r = c.get("/")
    assert r.status_code == 200 and "DocuMind" in r.text


def test_direct_answer_without_tools(make_client):
    c, llm, seen, _ = make_client([("text", "Hello! Ask me about your documents.")])
    r = c.post("/v1/chat", json={"message": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("Hello!")
    assert body["steps"] == 1 and body["citations"] == [] and seen == []


def test_tool_call_then_cited_answer(make_client):
    c, llm, seen, _ = make_client([SEARCH, ("text", "Risks above USD 250 million must be referred [S1].")])
    r = c.post("/v1/chat", json={"message": "When must I refer a risk?"}, headers={"X-Request-ID": "rid-42"})
    body = r.json()
    assert body["steps"] == 2
    assert [cit["source_id"] for cit in body["citations"]] == ["S1"]
    assert len(body["sources"]) == 2                    # both retrieved, one cited
    assert body["trace"][0]["tool"] == "search_knowledge_base" and body["trace"][0]["status"] == "ok"
    assert body["usage"] == {"prompt_tokens": 200, "completion_tokens": 40}
    # the request id travelled to knowledge-service
    assert seen[0].headers["x-request-id"] == "rid-42"
    # the second LLM call saw the assistant tool_call message and the tool result
    second = llm.calls[1]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
    assert second[-1]["role"] == "tool" and "S1" in second[-1]["content"]


def test_parallel_tool_calls(make_client):
    c, llm, *_ = make_client([
        ("tools", [("search_knowledge_base", {"query": "TIV"}), ("calculator", {"expression": "40000000*0.02"})]),
        ("text", "The deductible is 800,000 [S1]."),
    ])
    body = c.post("/v1/chat", json={"message": "q"}).json()
    tools = {t["tool"]: t for t in body["trace"]}
    assert tools["calculator"]["summary"] == "40000000*0.02 = 800000.0"
    assert tools["search_knowledge_base"]["status"] == "ok"


def test_memory_carries_over_between_turns(make_client):
    c, llm, *_ = make_client([("text", "First answer."), ("text", "Second answer.")])
    sid = c.post("/v1/chat", json={"message": "first question"}).json()["session_id"]
    c.post("/v1/chat", json={"message": "follow up", "session_id": sid})
    second_prompt = llm.calls[1]
    assert [m["content"] for m in second_prompt[1:]] == ["first question", "First answer.", "follow up"]
    history = c.get(f"/v1/sessions/{sid}").json()["messages"]
    assert len(history) == 4
    assert c.delete(f"/v1/sessions/{sid}").status_code == 204
    assert c.get(f"/v1/sessions/{sid}").json()["messages"] == []


def test_history_is_trimmed(make_client):
    c, llm, *_ = make_client([("text", f"a{i}") for i in range(5)])
    sid = c.post("/v1/chat", json={"message": "q0"}).json()["session_id"]
    for i in range(1, 5):
        c.post("/v1/chat", json={"message": f"q{i}", "session_id": sid})
    assert len(c.get(f"/v1/sessions/{sid}").json()["messages"]) == 4    # history_max_messages


def test_knowledge_service_down_degrades_gracefully(make_client):
    c, llm, seen, _ = make_client([SEARCH, ("text", "I cannot search the documents right now.")],
                                  knowledge_fail=True)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["trace"][0]["status"] == "error"
    assert "unavailable" in body["trace"][0]["summary"]
    assert len(seen) == 3                                   # 1 try + 2 retries
    assert "unavailable" in llm.calls[1][-1]["content"]     # the model was told


def test_invalid_tool_arguments_are_reported_to_model(make_client):
    c, llm, *_ = make_client([("tools", [("search_knowledge_base", "{not json")]), ("text", "ok")])
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["trace"][0]["status"] == "invalid"
    assert "not valid JSON" in llm.calls[1][-1]["content"]


def test_unknown_tool(make_client):
    c, llm, *_ = make_client([("tools", [("delete_everything", {})]), ("text", "ok")])
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["trace"][0]["status"] == "unknown"


def test_max_steps_forces_final_answer(make_client):
    c, llm, *_ = make_client([SEARCH, SEARCH, SEARCH, ("text", "final")], agent_max_steps=4)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["answer"] == "final" and body["steps"] == 4
    assert llm.tools_offered == [True, True, True, False]   # tools withheld on the last step


def test_llm_error_becomes_502(make_client):
    c, *_ = make_client([("error", LLMError("OpenRouter rejected the API key (401).", 401))])
    r = c.post("/v1/chat", json={"message": "q"})
    assert r.status_code == 502 and "API key" in r.json()["detail"]


def test_streaming_endpoint(make_client):
    c, *_ = make_client([SEARCH, ("text", "Refer above USD 250 million [S1].")])
    r = c.post("/v1/chat/stream", json={"message": "q"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no"
    events = _sse_events(r.text)
    types = [t for t, _ in events]
    assert types[0] == "session" and types[-1] == "done"
    assert "step" in types and "token" in types
    tokens = "".join(d["text"] for t, d in events if t == "token")
    assert "250 million" in tokens
    assert events[-1][1]["citations"][0]["filename"] == "underwriting-guidelines.md"


def test_stream_reports_llm_error_as_event(make_client):
    c, *_ = make_client([("error", LLMError("rate limit (429)", 429))])
    events = _sse_events(c.post("/v1/chat/stream", json={"message": "q"}).text)
    assert events[-1][0] == "error" and "429" in events[-1][1]["message"]


def test_rate_limit(make_client):
    c, *_ = make_client([("text", "ok")] * 5, rate_limit_per_minute=2)
    codes = [c.post("/v1/chat", json={"message": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_validation(make_client):
    c, *_ = make_client([])
    assert c.post("/v1/chat", json={"message": ""}).status_code == 422
    assert c.post("/v1/chat", json={"message": "x", "session_id": "bad id!"}).status_code == 422
    assert c.get("/v1/sessions/../../etc").status_code in (404, 422)


def test_bff_proxy(make_client):
    c, *_ = make_client([])
    assert c.get("/api/knowledge/v1/documents").json()[0]["filename"] == "underwriting-guidelines.md"
    r = c.post("/api/knowledge/v1/documents", files={"file": ("x.md", b"# hi", "text/markdown")})
    assert r.status_code == 201 and r.json()["status"] == "indexed"


def test_readyz_stays_ready_but_degraded_without_redis(make_client):
    c, llm, seen, r = make_client([])
    r.fake_server.connected = False             # fakeredis: simulate Redis outage
    res = c.get("/readyz")
    assert res.status_code == 200
    assert "redis" in res.json()["degraded"][0]
    # ...and chat still works (memory and rate limit fail open)
    llm.script.append(("text", "still answering"))
    assert c.post("/v1/chat", json={"message": "q"}).json()["answer"] == "still answering"


def test_readyz_fails_without_api_key(settings):
    import fakeredis
    from fastapi.testclient import TestClient
    from app.main import create_app
    from pydantic import SecretStr
    s = settings.model_copy(update={"openrouter_api_key": SecretStr("")})
    app = create_app(s, redis_client=fakeredis.FakeAsyncRedis(decode_responses=True))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and "OPENROUTER_API_KEY" in r.json()["problems"][0]


def test_metrics(make_client):
    c, *_ = make_client([SEARCH, ("text", "a [S1]")])
    c.post("/v1/chat", json={"message": "q"})
    text = c.get("/metrics").text
    for name in ("documind_llm_tokens_total", "documind_tool_calls_total",
                 "documind_agent_steps", 'route="/v1/chat"'):
        assert name in text


def test_options_endpoint_merges_both_services(make_client):
    c, *_ = make_client([])
    o = c.get("/v1/options").json()
    assert o["agent"]["default_llm_model"] == "openai/gpt-4o-mini"
    assert "openai/gpt-4.1-mini" in o["agent"]["llm_models"]
    assert o["knowledge"]["default_embedding_model"] == "openai/text-embedding-3-small"


def test_chat_options_reach_llm_and_search(make_client):
    c, llm, seen, _ = make_client([SEARCH, ("text", "ok [S1]")])
    body = c.post("/v1/chat", json={"message": "q", "options": {
        "llm_model": "openai/gpt-4.1-mini", "temperature": 0.7, "max_tokens": 300, "top_k": 2,
        "score_threshold": 0.4, "embedding_model": "openai/text-embedding-3-large", "max_steps": 3,
        "prompt_version": "v2"}}).json()
    assert llm.params[0] == {"model": "openai/gpt-4.1-mini", "temperature": 0.7, "max_tokens": 300}
    search_body = json.loads(seen[0].content)
    assert search_body["top_k"] == 2 and search_body["score_threshold"] == 0.4
    assert search_body["embedding_model"] == "openai/text-embedding-3-large"
    assert body["settings"]["prompt_version"] == "v2" and body["settings"]["max_steps"] == 3
    assert "under 120 words" in llm.calls[0][0]["content"]          # prompt v2 was used


def test_chat_options_are_validated(make_client):
    c, *_ = make_client([("text", "x")] * 5)
    bad = [{"llm_model": "someone/very-expensive-model"}, {"max_tokens": 50000}, {"max_steps": 50},
           {"top_k": 99}, {"temperature": 3}, {"prompt_version": "v9"}]
    for opts in bad:
        r = c.post("/v1/chat", json={"message": "q", "options": opts})
        assert r.status_code == 422, opts


def test_bff_passes_ingestion_settings(make_client):
    c, llm, seen, _ = make_client([])
    c.post("/api/knowledge/v1/documents?chunk_size=500&chunk_overlap=50&embedding_model=m",
           files={"file": ("x.md", b"# hi", "text/markdown")})
    q = dict(seen[-1].url.params)
    assert q == {"chunk_size": "500", "chunk_overlap": "50", "embedding_model": "m"}
