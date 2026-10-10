"""End-to-end tests of the multi-agent graph through the HTTP API.

Each test scripts the models per role (supervisor, grader, writer, ...) and so
drives one exact path through the graph."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph.state import Grades, Grounding, Memories, Plan, Rewrite, RouteDecision
from app.models import LLMError

from conftest import DOC_ID, prompt_text, searches, sse_events


def R(next_, instruction="", reason=""):
    return RouteDecision(next=next_, instruction=instruction, reason=reason)


RESEARCH_THEN_WRITE = {
    "supervisor": [R("researcher", "TIV referral threshold"), R("writer", reason="enough")],
    "grader": [Grades(relevant=[1])],
    "writer": ["Risks above USD 250 million must be referred [S1]."],
    "grounding": [Grounding(grounded=True)],
}


def actions(body):
    return [(t["agent"], t["action"]) for t in body["trace"]]


# ---------------------------------------------------------------------------- basics
def test_health_ready_info_and_ui(make_client):
    c, *_ = make_client()
    assert c.get("/healthz").json() == {"status": "ok"}
    assert c.get("/readyz").json() == {"status": "ready", "degraded": []}
    import socket
    assert c.get("/v1/info").headers["x-served-by"] == socket.gethostname()   # = pod name in k8s
    info = c.get("/v1/info").json()
    assert info["engine"] == "langgraph" and info["persistence"] == "memory"
    assert "supervisor" in info["agents"] and info["version"] == "2.0.0"
    r = c.get("/")
    assert r.status_code == 200 and "DocuMind" in r.text


def test_graph_endpoint_draws_all_agents(make_client):
    c, *_ = make_client()
    mermaid = c.get("/v1/graph").json()["mermaid"]
    for name in ("supervisor", "planner", "research_task", "librarian", "analyst", "grounding_check"):
        assert name in mermaid


# ---------------------------------------------------------------------------- routing
def test_direct_answer_skips_research(make_client):
    c, hub, seen, _ = make_client({"supervisor": [R("writer", reason="greeting")],
                                   "writer": ["Hello! Ask me about your documents."]})
    body = c.post("/v1/chat", json={"message": "hi"}).json()
    assert body["answer"].startswith("Hello!")
    assert body["steps"] == 1 and body["citations"] == [] and body["grounded"] is None
    assert searches(seen) == []
    assert hub.prompts("grounding") == []              # nothing to check without sources


def test_research_then_cited_answer(make_client):
    c, hub, seen, _ = make_client(RESEARCH_THEN_WRITE)
    r = c.post("/v1/chat", json={"message": "When must I refer a risk?"}, headers={"X-Request-ID": "rid-42"})
    body = r.json()
    assert r.status_code == 200, body
    assert [s["source_id"] for s in body["citations"]] == ["S1"]
    assert body["citations"][0]["filename"] == "underwriting-guidelines.md"
    assert len(body["sources"]) == 1                   # the grader dropped the irrelevant passage
    assert body["grounded"] is True and body["steps"] == 2
    assert ("researcher", "search") in actions(body) and ("researcher", "grade") in actions(body)
    assert searches(seen)[0]["query"] == "TIV referral threshold"     # the supervisor's instruction
    assert seen[0].headers["x-request-id"] == "rid-42"
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}
    assert "[S1] underwriting-guidelines.md" in prompt_text(hub.prompts("writer")[0])
    assert body["findings"][0]["agent"] == "researcher"


def test_corrective_rag_rewrites_the_query(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "referral rule"), R("writer")],
        "grader": [Grades(relevant=[]), Grades(relevant=[1])],
        "rewriter": [Rewrite(query="chief underwriter referral TIV")],
        "writer": ["Refer above USD 250 million [S1]."],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert [s["query"] for s in searches(seen)] == ["referral rule", "chief underwriter referral TIV"]
    assert ("researcher", "rewrite") in actions(body)
    assert body["citations"][0]["source_id"] == "S1"


def test_corrective_rag_gives_up_after_max_rewrites(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "flood cover"), R("writer")],
        "grader": [Grades(relevant=[]), Grades(relevant=[])],
        "rewriter": [Rewrite(query="flood exclusion")],
        "writer": ["The documents do not cover flood insurance."],
    }, max_query_rewrites=1)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert len(searches(seen)) == 2
    assert body["sources"] == [] and body["grounded"] is None
    assert "no relevant passages" in body["findings"][0]["result"]


def test_planner_researches_sub_questions_in_parallel(make_client):
    subs = ["How fast is a first notice of loss acknowledged?", "Who handles claims above USD 1 million?"]
    c, hub, seen, _ = make_client({
        "supervisor": [R("planner", "FNOL speed and large claims"), R("writer")],
        "planner": [Plan(sub_questions=subs)],
        "grader": [Grades(relevant=[1, 2]), Grades(relevant=[1, 2])],
        "writer": ["Within 24 hours [S2]; large claims go to the chief underwriter [S1]."],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert sorted(s["query"] for s in searches(seen)) == sorted(subs)
    assert body["plan"] == subs
    assert [f["agent"] for f in body["findings"]].count("researcher") == 2
    assert {s["source_id"] for s in body["citations"]} == {"S1", "S2"}


def test_analyst_uses_the_calculator_tool(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "deductible rule"), R("analyst", "2% of USD 40,000,000"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "analyst": [("tools", [("calculator", {"expression": "40000000*0.02"})]),
                    "The deductible is 800,000 (40,000,000 x 2%)."],
        "writer": ["The deductible is USD 800,000 [S1]."],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    calc = [t for t in body["trace"] if t["action"] == "calculator"]
    assert calc and calc[0]["detail"] == "40000000*0.02 = 800000.0"
    analyst = [f for f in body["findings"] if f["agent"] == "analyst"][0]
    assert "800,000" in analyst["result"]
    first, second = hub.prompts("analyst")
    assert "Instruction: 2% of USD 40,000,000" in prompt_text(first)
    assert isinstance(second[-1], ToolMessage) and "800000.0" in second[-1].content


def test_supervisor_step_limit_forces_the_writer(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "a"), R("researcher", "b"), R("researcher", "c")],
        "grader": [Grades(relevant=[1]), Grades(relevant=[1])],
        "writer": ["final [S1]"],
    })
    body = c.post("/v1/chat", json={"message": "q", "options": {"max_steps": 2}}).json()
    assert body["answer"] == "final [S1]" and body["steps"] == 2
    assert len(hub.prompts("supervisor")) == 2                # 3rd decision never asked
    assert any("step limit" in t["detail"] for t in body["trace"])


def test_loop_guard_stops_repeated_tasks(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "same query"), R("researcher", "same query")],
        "grader": [Grades(relevant=[1])],
        "writer": ["ok [S1]"],
    })
    c.post("/v1/chat", json={"message": "q"})
    assert len(searches(seen)) == 1


# ---------------------------------------------------------------------------- self-RAG
def test_grounding_check_sends_the_answer_back_for_revision(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "referral"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "writer": ["Risks above USD 100 million go to the CEO [S1].",
                   "Risks above USD 250 million go to the chief underwriter [S1]."],
        "grounding": [Grounding(grounded=False, unsupported_claims=["USD 100 million", "the CEO"]),
                      Grounding(grounded=True)],
    })
    events = sse_events(c.post("/v1/chat/stream", json={"message": "q"}).text)
    drafts = {d["draft"] for t, d in events if t == "token"}
    assert drafts == {1, 2}                                  # the UI resets on the new draft
    done = events[-1][1]
    assert done["answer"].startswith("Risks above USD 250 million") and done["drafts"] == 2
    assert done["grounded"] is True
    second = prompt_text(hub.prompts("writer")[1])
    assert "NOT supported" in second and "the CEO" in second


def test_ungrounded_answer_is_kept_after_max_revisions(make_client):
    c, hub, *_ = make_client({
        "supervisor": [R("researcher", "x"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "writer": ["draft one [S1]", "draft two [S1]"],
        "grounding": [Grounding(grounded=False, unsupported_claims=["a"]),
                      Grounding(grounded=False, unsupported_claims=["b"])],
    }, max_revisions=1)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["answer"] == "draft two [S1]" and body["grounded"] is False
    assert len(hub.prompts("writer")) == 2


# ---------------------------------------------------------------------------- human in the loop
LIBRARIAN_DELETE = {
    "supervisor": [R("librarian", "delete underwriting-guidelines.md"), R("writer")],
    "librarian": [("tools", [("list_documents", {})]),
                  ("tools", [("delete_document", {"doc_id": DOC_ID, "filename": "underwriting-guidelines.md"})])],
}


def test_delete_pauses_for_approval_then_runs(make_client):
    script = {**LIBRARIAN_DELETE,
              "librarian": [*LIBRARIAN_DELETE["librarian"], "Deleted underwriting-guidelines.md."],
              "writer": ["I deleted underwriting-guidelines.md."]}
    c, hub, seen, _ = make_client(script)
    body = c.post("/v1/chat", json={"message": "delete the underwriting guidelines"}).json()
    assert body["status"] == "interrupted"
    action = body["interrupt"]["actions"][0]
    assert action["name"] == "delete_document" and action["args"]["doc_id"] == DOC_ID
    assert set(body["interrupt"]["allowed"]) == {"approve", "edit", "reject"}
    assert not [r for r in seen if r.method == "DELETE"]          # nothing deleted yet

    done = c.post("/v1/chat/resume", json={"session_id": body["session_id"], "decision": "approve"}).json()
    assert done["status"] == "done" and done["answer"] == "I deleted underwriting-guidelines.md."
    assert [r.url.path for r in seen if r.method == "DELETE"] == [f"/v1/documents/{DOC_ID}"]
    assert c.post("/v1/chat/resume", json={"session_id": body["session_id"],
                                            "decision": "approve"}).status_code == 409


def test_rejected_delete_never_runs(make_client):
    script = {**LIBRARIAN_DELETE,
              "librarian": [*LIBRARIAN_DELETE["librarian"], "OK, I did not delete anything."],
              "writer": ["Nothing was deleted."]}
    c, hub, seen, _ = make_client(script)
    sid = c.post("/v1/chat", json={"message": "delete it"}).json()["session_id"]
    events = sse_events(c.post("/v1/chat/resume/stream", json={
        "session_id": sid, "decision": "reject", "message": "keep it"}).text)
    assert events[-1][0] == "done" and events[-1][1]["answer"] == "Nothing was deleted."
    assert not [r for r in seen if r.method == "DELETE"]
    last_prompt = hub.prompts("librarian")[-1]
    assert isinstance(last_prompt[-1], ToolMessage) and "keep it" in last_prompt[-1].content


def test_resume_without_pending_interrupt_is_409(make_client):
    c, *_ = make_client()
    r = c.post("/v1/chat/resume", json={"session_id": "nothing-here-123", "decision": "approve"})
    assert r.status_code == 409


def test_interrupt_is_streamed(make_client):
    c, *_ = make_client(LIBRARIAN_DELETE)
    events = sse_events(c.post("/v1/chat/stream", json={"message": "delete it"}).text)
    assert events[-1][0] == "interrupt"
    assert events[-1][1]["actions"][0]["args"]["filename"] == "underwriting-guidelines.md"


# ---------------------------------------------------------------------------- memory
def test_checkpointer_keeps_the_conversation(make_client):
    c, hub, *_ = make_client({"supervisor": [R("writer"), R("writer")],
                              "writer": ["First answer.", "Second answer."]})
    sid = c.post("/v1/chat", json={"message": "first question"}).json()["session_id"]
    c.post("/v1/chat", json={"message": "follow up", "session_id": sid})
    second = hub.prompts("writer")[1]
    assert [type(x) for x in second[1:3]] == [HumanMessage, AIMessage]
    assert [x.content for x in second[1:3]] == ["first question", "First answer."]
    history = c.get(f"/v1/sessions/{sid}").json()["messages"]
    assert [h["role"] for h in history] == ["user", "assistant", "user", "assistant"]
    assert c.delete(f"/v1/sessions/{sid}").status_code == 204
    assert c.get(f"/v1/sessions/{sid}").json()["messages"] == []


def test_history_sent_to_the_model_is_trimmed(make_client):
    c, hub, *_ = make_client({"supervisor": [R("writer")] * 4, "writer": ["a0", "a1", "a2", "a3"]})
    sid = c.post("/v1/chat", json={"message": "q0"}).json()["session_id"]
    for i in range(1, 4):
        c.post("/v1/chat", json={"message": f"q{i}", "session_id": sid})
    last = hub.prompts("writer")[-1]
    assert len(last) == 1 + 4 + 1                            # system + history_max_messages + question
    assert len(c.get(f"/v1/sessions/{sid}").json()["messages"]) == 8   # nothing lost in the checkpoint


def test_long_term_memory_crosses_sessions(make_client):
    c, hub, *_ = make_client({
        "supervisor": [R("writer"), R("writer")],
        "writer": ["Noted.", "Hello again."],
        "memory": [Memories(facts=["User is an underwriter", "Prefers short answers"])],
    })
    user = {"user_id": "user-12345678"}
    first = c.post("/v1/chat", json={"message": "I am an underwriter, keep answers short.", **user}).json()
    assert ("memory", "save") in actions(first)
    second = c.post("/v1/chat", json={"message": "hello", **user}).json()      # NEW session
    assert second["session_id"] != first["session_id"]
    assert ("memory", "recall") in actions(second)
    assert "Prefers short answers" in hub.prompts("writer")[1][0].content
    assert "User is an underwriter" in hub.prompts("supervisor")[1][0].content


def test_memory_extraction_is_skipped_when_the_user_says_nothing_about_themselves(make_client):
    c, hub, *_ = make_client({"supervisor": [R("writer")], "writer": ["ok"]})
    c.post("/v1/chat", json={"message": "What is the TIV limit?", "user_id": "user-12345678"})
    assert hub.prompts("memory") == []


# ---------------------------------------------------------------------------- time travel
def test_checkpoints_and_replay_fork_the_thread(make_client):
    c, hub, *_ = make_client({"supervisor": [R("writer")], "writer": ["first draft", "replayed answer"]})
    sid = c.post("/v1/chat", json={"message": "hello"}).json()["session_id"]
    cps = c.get(f"/v1/sessions/{sid}/checkpoints").json()["checkpoints"]
    assert cps and cps[0]["next"] == []                       # newest first: the finished turn
    before_writer = next(cp for cp in cps if cp["next"] == ["writer"])
    events = sse_events(c.post(f"/v1/sessions/{sid}/replay/stream",
                               json={"checkpoint_id": before_writer["checkpoint_id"]}).text)
    assert events[0][1]["mode"] == "replay"
    assert events[-1][0] == "done" and events[-1][1]["answer"] == "replayed answer"
    history = c.get(f"/v1/sessions/{sid}").json()["messages"]
    assert [h["content"] for h in history] == ["hello", "replayed answer"]      # the fork is now current
    assert c.post(f"/v1/sessions/{sid}/replay/stream", json={"checkpoint_id": "nope"}).status_code == 404


# ---------------------------------------------------------------------------- failures
def test_knowledge_service_down_degrades_gracefully(make_client):
    c, hub, seen, _ = make_client({"supervisor": [R("researcher", "x"), R("writer")],
                                   "writer": ["I cannot search the documents right now."]},
                                  knowledge_fail=True)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["answer"].startswith("I cannot search")
    assert "search failed" in body["findings"][0]["result"]
    assert len(seen) == 3                                     # 1 try + 2 retries
    assert any(t["status"] == "error" for t in body["trace"])


def test_llm_error_becomes_502(make_client):
    c, *_ = make_client({"supervisor": [LLMError("OpenRouter rejected the API key (401).", 401)]})
    r = c.post("/v1/chat", json={"message": "q"})
    assert r.status_code == 502 and "API key" in r.json()["detail"]


def test_stream_reports_llm_error_as_event(make_client):
    c, *_ = make_client({"supervisor": [R("writer")], "writer": [LLMError("rate limit (429)", 429)]})
    events = sse_events(c.post("/v1/chat/stream", json={"message": "q"}).text)
    assert events[-1][0] == "error" and "429" in events[-1][1]["message"]


def test_unparseable_decision_falls_back_to_writer(make_client):
    c, hub, seen, _ = make_client({"supervisor": [], "writer": ["fallback answer"]})   # returns None
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["answer"] == "fallback answer" and searches(seen) == []


# ---------------------------------------------------------------------------- streaming
def test_streaming_endpoint(make_client):
    c, *_ = make_client(RESEARCH_THEN_WRITE)
    r = c.post("/v1/chat/stream", json={"message": "q"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no"
    events = sse_events(r.text)
    types = [t for t, _ in events]
    assert types[0] == "session" and types[-1] == "done"
    assert events[0][1]["engine"] == "langgraph"
    assert "step" in types and "token" in types
    tokens = "".join(d["text"] for t, d in events if t == "token")
    assert "250 million" in tokens                            # only the writer's tokens are streamed
    assert events[-1][1]["citations"][0]["filename"] == "underwriting-guidelines.md"


# ---------------------------------------------------------------------------- API guardrails
def test_rate_limit(make_client):
    c, *_ = make_client({"supervisor": [R("writer")] * 5, "writer": ["ok"] * 5}, rate_limit_per_minute=2)
    codes = [c.post("/v1/chat", json={"message": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_validation(make_client):
    c, *_ = make_client()
    assert c.post("/v1/chat", json={"message": ""}).status_code == 422
    assert c.post("/v1/chat", json={"message": "x", "session_id": "bad id!"}).status_code == 422
    assert c.post("/v1/chat", json={"message": "x", "user_id": "no"}).status_code == 422
    assert c.post("/v1/chat/resume", json={"session_id": "abcdefgh1", "decision": "maybe"}).status_code == 422
    assert c.get("/v1/sessions/../../etc").status_code in (404, 422)


def test_chat_options_reach_models_and_search(make_client):
    c, hub, seen, _ = make_client(RESEARCH_THEN_WRITE)
    body = c.post("/v1/chat", json={"message": "q", "options": {
        "llm_model": "openai/gpt-4.1-mini", "temperature": 0.7, "max_tokens": 300, "top_k": 2,
        "score_threshold": 0.4, "embedding_model": "openai/text-embedding-3-large", "max_steps": 3,
        "prompt_version": "v2"}}).json()
    writer = [p for p in hub.params if p["role"] == "writer"][0]
    assert writer == {"role": "writer", "model": "openai/gpt-4.1-mini", "temperature": 0.7, "max_tokens": 300}
    search_body = searches(seen)[0]
    assert search_body["top_k"] == 2 and search_body["score_threshold"] == 0.4
    assert search_body["embedding_model"] == "openai/text-embedding-3-large"
    assert body["settings"]["prompt_version"] == "v2" and body["settings"]["max_steps"] == 3
    assert "under 120 words" in hub.prompts("writer")[0][0].content       # prompt v2 was used


def test_chat_options_are_validated(make_client):
    c, *_ = make_client()
    bad = [{"llm_model": "someone/very-expensive-model"}, {"max_tokens": 50000}, {"max_steps": 50},
           {"top_k": 99}, {"temperature": 3}, {"prompt_version": "v9"}]
    for opts in bad:
        r = c.post("/v1/chat", json={"message": "q", "options": opts})
        assert r.status_code == 422, opts


def test_options_endpoint_merges_both_services(make_client):
    c, *_ = make_client()
    o = c.get("/v1/options").json()
    assert o["agent"]["default_llm_model"] == "openai/gpt-4o-mini"
    assert o["knowledge"]["default_embedding_model"] == "openai/text-embedding-3-small"


def test_bff_proxy(make_client):
    c, llm, seen, _ = make_client()
    assert c.get("/api/knowledge/v1/documents").json()[0]["filename"] == "underwriting-guidelines.md"
    r = c.post("/api/knowledge/v1/documents?chunk_size=500&chunk_overlap=50&embedding_model=m",
               files={"file": ("x.md", b"# hi", "text/markdown")})
    assert r.status_code == 201 and r.json()["status"] == "indexed"
    assert dict(seen[-1].url.params) == {"chunk_size": "500", "chunk_overlap": "50", "embedding_model": "m"}


def test_readyz_stays_ready_but_degraded_without_redis(make_client):
    c, hub, seen, r = make_client({"supervisor": [R("writer")], "writer": ["still answering"]})
    r.fake_server.connected = False             # fakeredis: simulate Redis outage
    res = c.get("/readyz")
    assert res.status_code == 200 and "redis" in res.json()["degraded"][0]
    assert c.post("/v1/chat", json={"message": "q"}).json()["answer"] == "still answering"


def test_readyz_fails_without_api_key(settings):
    import fakeredis
    from fastapi.testclient import TestClient
    from pydantic import SecretStr

    from app.main import create_app
    from app.persistence import in_memory
    s = settings.model_copy(update={"openrouter_api_key": SecretStr("")})
    app = create_app(s, redis_client=fakeredis.FakeAsyncRedis(decode_responses=True), persistence=in_memory())
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and "OPENROUTER_API_KEY" in r.json()["problems"][0]


def test_metrics(make_client):
    c, *_ = make_client(RESEARCH_THEN_WRITE)
    c.post("/v1/chat", json={"message": "q"})
    text = c.get("/metrics").text
    for name in ("documind_llm_tokens_total", "documind_graph_node_runs_total",
                 'documind_supervisor_routes_total{next="researcher"}', "documind_grounding_checks_total",
                 "documind_agent_steps", 'route="/v1/chat"'):
        assert name in text, name


def test_claim_level_check_quotes_the_contradicting_sentence(make_client):
    from app.graph.state import ClaimCheck
    wrong = ClaimCheck(claim="Cover ends when the surrender value is paid", source_id="S1",
                       quote="Upon receipt of a valid surrender request the Insurance shall cease.",
                       supported=False)
    c, hub, *_ = make_client({
        "supervisor": [R("researcher", "surrender"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "writer": ["Cover ends when the surrender value is paid [S1].",
                   "Cover ends on receipt of a valid surrender request [S1]."],
        # the model said grounded=True, but one claim check failed: the claim check wins
        "grounding": [Grounding(claims=[wrong], grounded=True), Grounding(grounded=True)],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["drafts"] == 2 and body["answer"].startswith("Cover ends on receipt")
    feedback = prompt_text(hub.prompts("writer")[1])
    assert "the source says" in feedback and "Upon receipt of a valid surrender request" in feedback


def test_uncited_paragraph_sends_the_answer_back(make_client):
    c, hub, *_ = make_client({
        "supervisor": [R("researcher", "referral"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "writer": ["Risks above USD 250 million must be referred to the chief underwriter.\n\n"
                   "The documents do not cover flood risks.",
                   "Risks above USD 250 million must be referred to the chief underwriter [S1].\n\n"
                   "The documents do not cover flood risks."],
        "grounding": [Grounding(grounded=True), Grounding(grounded=True)],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["drafts"] == 2 and body["grounded"] is True
    feedback = prompt_text(hub.prompts("writer")[1])
    assert "no citation" in feedback and "flood" not in feedback.split("NOT supported")[1]
    assert any("uncited paragraphs" in t["detail"] for t in body["trace"])
