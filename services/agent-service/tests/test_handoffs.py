"""Communication between agents: every handoff that loses or degrades
information must be visible (issue in the answer + trace + metric)."""
from app.graph.common import format_findings, select_sources
from app.graph.decisions import similar_task
from app.graph.handoffs import sub_agent_report
from app.graph.state import Grades, Grounding, RouteDecision
from langchain_core.messages import AIMessage, ToolMessage

from conftest import PASSAGES, prompt_text


def R(next_, instruction="", reason=""):
    return RouteDecision(next=next_, instruction=instruction, reason=reason)


def kinds(body):
    return [(i["source"], i["target"], i["kind"]) for i in body["issues"]]


# ---------------------------------------------------------------------------- researcher -> supervisor
def test_findings_are_typed_with_status_and_data(make_client):
    c, *_ = make_client({"supervisor": [R("researcher", "referral"), R("writer")],
                         "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]})
    f = c.post("/v1/chat", json={"message": "q"}).json()["findings"][0]
    assert f["status"] == "ok"
    assert f["data"]["passages"] == 1 and f["data"]["pages"] == [1] and f["data"]["queries"] == ["referral"]


def test_nothing_relevant_is_an_empty_finding_and_an_issue(make_client):
    c, *_ = make_client({"supervisor": [R("researcher", "flood"), R("writer")],
                         "grader": [Grades(relevant=[]), Grades(relevant=[])],
                         "writer": ["The documents do not cover floods."]})
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["findings"][0]["status"] == "empty"
    assert ("researcher", "supervisor", "no_evidence") in kinds(body)


def test_search_failure_is_an_error_finding_and_an_issue(make_client):
    c, *_ = make_client({"supervisor": [R("researcher", "x"), R("writer")], "writer": ["Search is down."]},
                        knowledge_fail=True)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert body["findings"][0]["status"] == "error"
    assert ("researcher", "supervisor", "search_failed") in kinds(body)
    assert any(t["agent"] == "handoff" and t["status"] == "error" for t in body["trace"])


def test_supervisor_sees_excerpts_of_the_evidence(make_client):
    c, hub, *_ = make_client({"supervisor": [R("researcher", "referral"), R("writer")],
                              "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]})
    c.post("/v1/chat", json={"message": "q"})
    second = prompt_text(hub.prompts("supervisor")[1])
    assert "Risks above USD 250 million TIV must be referred" in second       # content, not just a count
    assert "[researcher | ok]" in second


def test_unusable_grader_verdict_is_reported(make_client):
    c, *_ = make_client({"supervisor": [R("researcher", "x"), R("writer")],
                         "grader": [{"relevant": "all"}, {"relevant": "all"}],     # invalid twice
                         "writer": ["ok [S1]"]})
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert ("grader", "researcher", "grader_fallback") in kinds(body)
    assert len(body["sources"]) == 2                                          # kept, but flagged


# ---------------------------------------------------------------------------- researcher -> writer
def test_truncation_before_the_writer_is_reported(make_client):
    c, hub, *_ = make_client({"supervisor": [R("researcher", "x"), R("writer")],
                              "grader": [Grades(relevant=[1, 2])], "writer": ["ok [S1]"],
                              "grounding": [Grounding(grounded=True)]},
                             tool_result_max_chars=80)
    body = c.post("/v1/chat", json={"message": "q"}).json()
    found = [i for i in body["issues"] if i["kind"] == "truncated"]
    assert found and "of 2 passages" in found[0]["detail"]
    assert "First notice of loss is acknowledged" not in prompt_text(hub.prompts("writer")[0])


def test_select_sources_counts_what_it_drops():
    srcs = {p["doc_id"]: {**p, "key": p["doc_id"], "query": "q"} for p in PASSAGES}
    full = select_sources(srcs, max_n=10, max_chars=10_000)
    assert (len(full.numbered), full.dropped, full.cut, full.lossy) == (2, 0, 0, False)
    one = select_sources(srcs, max_n=1, max_chars=10_000)
    assert (one.dropped, one.lossy) == (1, True) and "saw 1 of 2 passages" in one.describe()


# ---------------------------------------------------------------------------- supervisor -> writer
def test_the_writer_receives_the_supervisors_brief(make_client):
    c, hub, *_ = make_client({"supervisor": [R("researcher", "free look"),
                                             R("writer", "Give the period in days AND in hours")],
                              "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]})
    c.post("/v1/chat", json={"message": "q"})
    assert "Instruction from the supervisor for this answer: Give the period in days AND in hours" \
        in prompt_text(hub.prompts("writer")[0])


def test_a_stale_worker_instruction_is_never_the_writers_brief(make_client):
    c, hub, *_ = make_client({"supervisor": [R("researcher", "TIV threshold for referral")],
                              "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]},
                             agent_max_steps=1)                  # step limit forces the writer
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert "Instruction from the supervisor" not in prompt_text(hub.prompts("writer")[0])
    assert ("supervisor", "writer", "step_limit") in kinds(body)


# ---------------------------------------------------------------------------- sub-agent -> supervisor
def test_sub_agent_stopped_by_its_call_limit_is_a_partial_report(make_client):
    calls = [("tools", [("calculator", {"expression": f"{i}+1"})]) for i in range(6)]   # limit = 6
    c, hub, *_ = make_client({"supervisor": [R("analyst", "add things up"), R("writer")],
                              "analyst": calls, "writer": ["Could not finish the calculation."]})
    body = c.post("/v1/chat", json={"message": "q"}).json()
    analyst = [f for f in body["findings"] if f["agent"] == "analyst"][0]
    assert analyst["status"] == "partial" and analyst["result"].startswith("INCOMPLETE")
    assert "Model call limits exceeded" not in analyst["result"].split("Last tool results")[0]
    assert ("analyst", "supervisor", "partial_report") in kinds(body)


def test_sub_agent_report_reader():
    done = [AIMessage("", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}]),
            ToolMessage('{"result": 2}', tool_call_id="1"), AIMessage("The total is 2.")]
    assert sub_agent_report(done)[:2] == ("The total is 2.", "ok")
    stopped = [*done[:2], AIMessage("Model call limits exceeded: run limit (6/6)")]
    text, status, data = sub_agent_report(stopped)
    assert status == "partial" and '{"result": 2}' in text and data["stopped_by"].startswith("Model call")
    assert sub_agent_report([AIMessage("")])[1] == "partial"


# ---------------------------------------------------------------------------- supervisor -> worker
def test_empty_instruction_is_reported(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [{"next": "researcher", "instruction": ""}, {"next": "researcher", "instruction": ""},
                       R("writer")],                       # invalid twice -> lenient keeps it empty
        "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]})
    body = c.post("/v1/chat", json={"message": "What is the TIV limit?"}).json()
    assert ("supervisor", "researcher", "empty_instruction") in kinds(body)
    assert [r for r in seen if r.url.path == "/v1/search"]          # the worker used the question


def test_reworded_repeat_is_caught():
    assert similar_task("free look period distance marketing", "Free look period for distance marketing")
    assert not similar_task("free look period", "grace period for premium payment")
    assert not similar_task("TIV limit", "TIV threshold")          # synonyms: out of reach (documented)


def test_no_valid_decision_is_reported(make_client):
    c, *_ = make_client({"supervisor": [], "writer": ["fallback answer"]})
    body = c.post("/v1/chat", json={"message": "q"}).json()
    assert ("supervisor", "writer", "fallback_decision") in kinds(body)


# ---------------------------------------------------------------------------- inspection + compatibility
def test_handoffs_endpoint_replays_who_passed_what(make_client):
    c, *_ = make_client({"supervisor": [R("researcher", "referral rule"), R("writer", "be brief")],
                         "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]})
    sid = c.post("/v1/chat", json={"message": "q"}).json()["session_id"]
    steps = c.get(f"/v1/sessions/{sid}/handoffs").json()["handoffs"]
    research = next(s for s in steps if s["ran"] == ["researcher"])
    assert research["instruction"] == "referral rule"
    assert research["new_findings"][0]["agent"] == "researcher" and research["sources"] == 1
    assert next(s for s in steps if s["ran"] == ["writer"])["writer_brief"] == "be brief"


def test_findings_from_older_checkpoints_still_work():
    old = [{"agent": "researcher", "task": "x", "result": "3 relevant passages"}]      # no status/data
    assert "[researcher | ok]" in format_findings(old)


def test_handoff_metric(make_client):
    c, *_ = make_client({"supervisor": [], "writer": ["x"]})
    c.post("/v1/chat", json={"message": "q"})
    assert ('documind_handoff_issues_total{kind="fallback_decision",source="supervisor",target="writer"}'
            in c.get("/metrics").text)
