"""Advanced Pydantic in the agent: validators with context, self-correction,
lenient repair, discriminated unions, TypeAdapter contracts, Annotated tool limits."""
import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.events import EVENT_ADAPTER, DoneEvent, StepEvent, TokenEvent
from app.graph.decisions import Grades, Grounding, Plan, RouteDecision, quote_in_source
from app.graph.tools import calculator, delete_document

from conftest import DOC_ID, PASSAGES, knowledge_transport, prompt_text, sse_events

def R(next_, instruction="", reason=""):
    return RouteDecision(next=next_, instruction=instruction, reason=reason)


# ---------------------------------------------------------------------------- validators + context
def test_grades_range_is_checked_only_with_context():
    assert Grades(relevant=[3, 1, 3]).relevant == [1, 3]                 # no context: shape only
    with pytest.raises(ValidationError, match="between 1 and 2"):
        Grades.model_validate({"relevant": [1, 7]}, context={"n_passages": 2})
    repaired = Grades.model_validate({"relevant": [1, 7]}, context={"n_passages": 2, "lenient": True})
    assert repaired.relevant == [1]


def test_route_decision_needs_an_instruction_and_rejects_repeats():
    with pytest.raises(ValidationError, match="instruction is required"):
        RouteDecision.model_validate({"next": "researcher", "instruction": "  "})
    done = {("researcher", "tiv limit"): "3 relevant passages"}
    with pytest.raises(ValidationError, match="already did exactly this task"):
        RouteDecision.model_validate({"next": "researcher", "instruction": "TIV limit"}, context={"done": done})
    lenient = RouteDecision.model_validate({"next": "researcher", "instruction": "TIV limit"},
                                           context={"done": done, "lenient": True})
    assert lenient.next == "writer"


def test_plan_is_cleaned_and_bounded():
    p = Plan(sub_questions=["What is the TIV limit?", "what is the tiv limit?", " Who approves large claims? "])
    assert p.sub_questions == ["What is the TIV limit?", "Who approves large claims?"]
    with pytest.raises(ValidationError, match="3\\+ words"):
        Plan(sub_questions=["TIV?"])
    many = {"sub_questions": [f"What is rule number {i}?" for i in range(6)]}
    with pytest.raises(ValidationError, match="at most 4"):
        Plan.model_validate(many, context={"max_subquestions": 4})
    assert len(Plan.model_validate(many, context={"max_subquestions": 4, "lenient": True}).sub_questions) == 4


SOURCE = "3.3.2. Upon receipt of a valid surrender request from the Member, the Insurance in respect of such Member shall cease."


def test_quote_must_exist_in_the_cited_source():
    assert quote_in_source("upon receipt of a valid surrender request ... shall cease", SOURCE)
    assert quote_in_source("Upon receipt of a valid  surrender request", SOURCE)       # whitespace/case
    assert not quote_in_source("Upon payment of the Surrender Value the Insurance shall cease", SOURCE)
    # real PDF extraction artefacts from the policy document
    pdf = "a Member may at any time , request for the surrender ... (Unexpired months / Total Period of Cover age in months )"
    assert quote_in_source("a Member may at any time, request for the surrender", pdf)
    assert quote_in_source("Total Period of Coverage in months)", pdf)
    claim = {"claim": "Cover ends on payment", "source_id": "S1", "supported": True,
             "quote": "Upon payment of the Surrender Value the Insurance shall cease"}
    ctx = {"sources": {"S1": SOURCE}}
    with pytest.raises(ValidationError, match="not a verbatim sentence of S1"):
        Grounding.model_validate({"claims": [claim], "grounded": True}, context=ctx)
    with pytest.raises(ValidationError, match="not one of"):
        Grounding.model_validate({"claims": [{**claim, "source_id": "S9"}], "grounded": True}, context=ctx)
    repaired = Grounding.model_validate({"claims": [claim], "grounded": True}, context={**ctx, "lenient": True})
    assert repaired.grounded is False and "not a verbatim sentence" in repaired.problems()[0]


# ---------------------------------------------------------------------------- self-correction loop
def test_invalid_grades_are_sent_back_and_corrected(make_client):
    c, hub, *_ = make_client({
        "supervisor": [R("researcher", "referral"), R("writer")],
        "grader": [{"relevant": [1, 9]}, Grades(relevant=[1])],     # 1st: passage 9 does not exist
        "writer": ["Refer above USD 250 million [S1]."],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert [s["source_id"] for s in body["citations"]] == ["S1"]
    retry = hub.prompts("grader")[1]
    assert isinstance(retry[-1], ToolMessage) and "between 1 and 2" in retry[-1].content
    metrics = c.get("/metrics").text
    assert 'documind_structured_outputs_total{outcome="corrected",schema="Grades"}' in metrics


def test_still_invalid_after_retry_is_repaired_leniently(make_client):
    c, hub, *_ = make_client({
        "supervisor": [R("researcher", "referral"), R("writer")],
        "grader": [{"relevant": [1, 9]}, {"relevant": [1, 8]}],     # wrong twice -> drop 8
        "writer": ["Refer above USD 250 million [S1]."],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert len(body["sources"]) == 1
    assert 'outcome="repaired",schema="Grades"' in c.get("/metrics").text


def test_supervisor_is_told_a_task_was_already_done(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("researcher", "TIV limit"), R("researcher", "TIV limit"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "writer": ["ok [S1]"],
    })
    c.post("/v1/chat", json={"message": "q"})
    feedback = hub.prompts("supervisor")[2][-1]
    assert isinstance(feedback, ToolMessage) and "already did exactly this task" in feedback.content
    assert len([r for r in seen if r.url.path == "/v1/search"]) == 1


def test_made_up_quote_makes_the_checker_try_again(make_client):
    bad = {"claims": [{"claim": "Refer above 250m", "source_id": "S1", "quote": "Anything above 100m", "supported": True}],
           "grounded": True}
    good = {"claims": [{"claim": "Refer above 250m", "source_id": "S1", "supported": True,
                        "quote": "Risks above USD 250 million TIV must be referred to the chief underwriter."}],
            "grounded": True}
    c, hub, *_ = make_client({
        "supervisor": [R("researcher", "referral"), R("writer")],
        "grader": [Grades(relevant=[1])],
        "writer": ["Risks above USD 250 million must be referred [S1]."],
        "grounding": [bad, good],
    })
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["grounded"] is True and body["drafts"] == 1
    assert "not a verbatim sentence of S1" in hub.prompts("grounding")[1][-1].content


# ---------------------------------------------------------------------------- discriminated unions
def test_every_streamed_event_parses_into_the_union(make_client):
    c, *_ = make_client({"supervisor": [R("researcher", "x"), R("writer")], "grader": [Grades(relevant=[1])],
                         "writer": ["Refer above USD 250 million [S1]."]})
    raw = c.post("/v1/chat/stream", json={"message": "q"}).text
    events = [EVENT_ADAPTER.validate_python(data) for _, data in sse_events(raw)]
    assert isinstance(events[-1], DoneEvent) and events[-1].usage.prompt_tokens == 100
    assert any(isinstance(e, TokenEvent) for e in events) and any(isinstance(e, StepEvent) for e in events)
    assert EVENT_ADAPTER.validate_json('{"type": "token", "text": "Hi", "draft": 1}') == TokenEvent(text="Hi", draft=1)
    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python({"type": "token", "text": "Hi", "draft": 0})      # draft >= 1


def test_events_schema_endpoint(make_client):
    c, *_ = make_client()
    schema = c.get("/v1/events/schema").json()
    assert schema["discriminator"]["propertyName"] == "type"
    assert {"session", "step", "token", "interrupt", "done", "error"} == set(schema["discriminator"]["mapping"])


DELETE = {
    "supervisor": [R("librarian", "delete underwriting-guidelines.md"), R("writer")],
    "librarian": [("tools", [("delete_document", {"doc_id": DOC_ID, "filename": "underwriting-guidelines.md"})]),
                  "Done."],
    "writer": ["Handled."],
}


def test_edit_decision_is_validated_against_the_tool_schema(make_client):
    c, hub, seen, _ = make_client(DELETE)
    sid = c.post("/v1/chat", json={"message": "delete it"}).json()["session_id"]
    bad =c.post("/v1/chat/resume", json={"session_id": sid, "decision": "edit",
                                           "args": {"doc_id": "../../etc", "filename": "x"}})
    assert bad.status_code == 422 and "invalid arguments for delete_document" in bad.text
    other = "fedcba9876543210"
    ok = c.post("/v1/chat/resume", json={"session_id": sid, "decision": "edit",
                                          "args": {"doc_id": other, "filename": "other.md"}})
    assert ok.json()["status"] == "done"
    assert [r.url.path for r in seen if r.method == "DELETE"] == [f"/v1/documents/{other}"]   # edited call ran


def test_resume_union_rejects_fields_of_another_decision(make_client):
    c, *_ = make_client()
    r = c.post("/v1/chat/resume", json={"session_id": "abcdefgh1", "decision": "approve", "args": {}})
    assert r.status_code == 422                     # extra="forbid": args belong to "edit" only
    r = c.post("/v1/chat/resume", json={"session_id": "abcdefgh1", "decision": "edit"})
    assert r.status_code == 422 and "args" in r.text


# ---------------------------------------------------------------------------- tool limits
def test_tool_schemas_carry_the_limits():
    calc = calculator.tool_call_schema.model_json_schema()["properties"]["expression"]
    assert calc["maxLength"] == 200
    doc = delete_document.tool_call_schema.model_json_schema()["properties"]["doc_id"]
    assert doc["pattern"] == "^[0-9a-f]{16}$"


def test_invalid_tool_arguments_never_reach_knowledge_service(make_client):
    script = {**DELETE, "librarian": [("tools", [("delete_document", {"doc_id": "../../etc", "filename": "x"})]),
                                      "I could not delete it."]}
    c, hub, seen, _ = make_client(script)
    sid = c.post("/v1/chat", json={"message": "delete it"}).json()["session_id"]
    c.post("/v1/chat/resume", json={"session_id": sid, "decision": "approve"})
    assert not [r for r in seen if r.method == "DELETE"]
    tool_msg = hub.prompts("librarian")[-1][-1]
    assert isinstance(tool_msg, ToolMessage) and "should match pattern" in tool_msg.content


# ---------------------------------------------------------------------------- boundary contract
def test_malformed_knowledge_response_is_a_contract_error():
    import asyncio

    import httpx

    from app.knowledge_client import KnowledgeClient, KnowledgeContractError

    def handler(request):
        return httpx.Response(200, json={"results": [{**PASSAGES[0], "page": 0}]})     # page must be >= 1
    kc = KnowledgeClient("http://k", 5, transport=httpx.MockTransport(handler))
    with pytest.raises(KnowledgeContractError, match="results.0.page"):
        asyncio.run(kc.search("q", 3))
    hits = asyncio.run(KnowledgeClient("http://k", 5, transport=knowledge_transport()[0]).search("q", 2))
    assert hits[0].filename == "underwriting-guidelines.md" and hits[0].score == 0.82


def test_contract_error_degrades_gracefully_in_the_graph(make_client, monkeypatch):
    from app.knowledge_client import KnowledgeClient, KnowledgeContractError

    async def broken(self, *a, **k):
        raise KnowledgeContractError("knowledge-service returned an unexpected search response (results.0.page)")
    monkeypatch.setattr(KnowledgeClient, "search", broken)
    c, *_ = make_client({"supervisor": [R("researcher", "x"), R("writer")], "writer": ["Search is unavailable."]})
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert "unexpected search response" in body["findings"][0]["result"]


# ---------------------------------------------------------------------------- truncation + honesty
def test_cut_off_reply_is_explained_to_the_model():
    import asyncio

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda

    from app.graph.common import decide
    seen = []
    replies = [AIMessage("", tool_calls=[{"name": "Grades", "args": {"relevant": "1, 2, ..."}, "id": "c1"}],
                         response_metadata={"finish_reason": "length"}),
               AIMessage("", tool_calls=[{"name": "Grades", "args": {"relevant": [1]}, "id": "c2"}])]

    async def model(messages):
        seen.append(list(messages))
        return {"raw": replies.pop(0), "parsed": None, "parsing_error": None}
    result = asyncio.run(decide(RunnableLambda(model), [HumanMessage("grade")], Grades, None))
    assert result.relevant == [1]
    assert "CUT OFF at the output token limit" in seen[1][-1].content


def test_a_check_that_never_ran_is_not_reported_as_fact_checked(make_client):
    truncated = {"claims": [{"claim": "Refer above 250m", "source_id": "S1"}]}     # no supported, no grounded
    c, *_ = make_client({"supervisor": [R("researcher", "x"), R("writer")], "grader": [Grades(relevant=[1])],
                         "writer": ["Refer above USD 250 million [S1]."], "grounding": [truncated, truncated]})
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["grounded"] is None
    assert any("NOT verified" in t["detail"] for t in body["trace"])
    assert 'outcome="fallback",schema="Grounding"' in c.get("/metrics").text
