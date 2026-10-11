"""MISSED handoffs: something that should have been passed on never was.
Each test causes one on purpose and checks that it is reported."""
from langchain_core.messages import ToolMessage

from app.graph.nodes import EMPTY_ANSWER
from app.graph.state import RESET, Grades, RouteDecision, add_findings

from conftest import DOC_ID, prompt_text


def R(next_, instruction="", needs_documents=False):
    return RouteDecision(next=next_, instruction=instruction, needs_documents=needs_documents)


def kinds(body):
    return [i["kind"] for i in body.get("issues") or []]


# ---------------------------------------------------------------------------- gap 1: research skipped
def test_skipping_research_is_corrected_by_the_validator(make_client):
    c, hub, seen, _ = make_client({
        "supervisor": [R("writer", needs_documents=True),             # tries to skip the search
                       R("researcher", "surrender value formula", True), R("writer", needs_documents=True)],
        "grader": [Grades(relevant=[1])], "writer": ["ok [S1]"]})
    body = c.post("/v1/chat", json={"message": "What is the surrender value formula?"}).json()
    retry = hub.prompts("supervisor")[1][-1]
    assert isinstance(retry, ToolMessage) and "nothing has been searched yet" in retry.content
    assert [r for r in seen if r.url.path == "/v1/search"]             # the research did happen
    assert "no_research" not in kinds(body)


def test_research_still_skipped_is_reported(make_client):
    c, *_ = make_client({"supervisor": [R("writer", needs_documents=True)] * 2,   # insists twice
                         "writer": ["From general knowledge: ..."]})
    body = c.post("/v1/chat", json={"message": "What is the surrender value formula?"}).json()
    assert "no_research" in kinds(body)


def test_greeting_needs_no_research(make_client):
    c, *_ = make_client({"supervisor": [R("writer", needs_documents=False)], "writer": ["Hello!"]})
    assert "no_research" not in kinds(c.post("/v1/chat", json={"message": "hi"}).json())


# ---------------------------------------------------------------------------- gap 2: empty answer
def test_empty_answer_is_retried(make_client):
    c, hub, *_ = make_client({"supervisor": [R("writer")], "writer": ["", "Hello, how can I help?"]})
    body = c.post("/v1/chat", json={"message": "hi"}).json()
    assert body["answer"] == "Hello, how can I help?" and body["drafts"] == 2
    assert "empty_answer" in kinds(body)
    assert "Your previous draft was EMPTY" in prompt_text(hub.prompts("writer")[1])


def test_still_empty_gets_a_clear_message_not_a_blank(make_client):
    c, *_ = make_client({"supervisor": [R("writer")], "writer": ["", ""]}, max_revisions=1)
    body = c.post("/v1/chat", json={"message": "hi"}).json()
    assert body["answer"] == EMPTY_ANSWER and kinds(body).count("empty_answer") == 2


# ---------------------------------------------------------------------------- gap 3: run died midway
def test_a_run_that_died_midway_is_reported_on_the_next_message(make_client):
    c, *_ = make_client({"supervisor": [R("writer"), R("writer")],
                         "writer": [RuntimeError("pod killed"), "Second answer."]})
    sid = "crash-test-0001"
    assert c.post("/v1/chat", json={"message": "first", "session_id": sid}).status_code == 502
    stuck = c.get("/v1/threads/stuck?older_than_s=0").json()
    assert [t["session_id"] for t in stuck["abandoned"]] == [sid]
    assert stuck["abandoned"][0]["next"] == ["writer"]
    body = c.post("/v1/chat", json={"message": "second", "session_id": sid}).json()
    assert body["answer"] == "Second answer."
    lost = [i for i in body["issues"] if i["kind"] == "abandoned_run"]
    assert lost and "writer" in lost[0]["detail"]
    assert c.get("/v1/threads/stuck?older_than_s=0").json()["abandoned"] == []      # healed


# ---------------------------------------------------------------------------- gap 4: unanswered approval
DELETE = {"supervisor": [R("librarian", "delete underwriting-guidelines.md"), R("writer")],
          "librarian": [("tools", [("delete_document", {"doc_id": DOC_ID, "filename": "underwriting-guidelines.md"})])],
          "writer": ["Hello again."]}


def test_unanswered_approvals_are_listed_and_counted(make_client):
    c, *_ = make_client(DELETE)
    sid = c.post("/v1/chat", json={"message": "delete it"}).json()["session_id"]
    stuck = c.get("/v1/threads/stuck?older_than_s=0").json()
    assert stuck["pending_approval"][0]["session_id"] == sid
    assert stuck["pending_approval"][0]["actions"] == ["delete_document"]
    assert c.get("/v1/threads/stuck").json()["pending_approval"] == []    # not older than STUCK_AFTER_S yet
    c.get("/v1/threads/stuck?older_than_s=0")
    assert 'documind_stuck_conversations{state="pending_approval"} 1.0' in c.get("/metrics").text


def test_new_message_instead_of_a_decision_is_reported(make_client):
    c, hub, seen, _ = make_client(DELETE)
    sid = c.post("/v1/chat", json={"message": "delete it"}).json()["session_id"]
    hub.models["supervisor"].script.append(R("writer"))
    body = c.post("/v1/chat", json={"message": "never mind, hello", "session_id": sid}).json()
    dropped = [i for i in body["issues"] if i["kind"] == "approval_abandoned"]
    assert dropped and "delete_document" in dropped[0]["detail"]
    assert not [r for r in seen if r.method == "DELETE"]                   # nothing was deleted


# ---------------------------------------------------------------------------- the reducer that carries them
def test_reset_and_start_with_carried_items():
    assert add_findings([{"a": 1}], [RESET, {"b": 2}]) == [{"b": 2}]
    assert add_findings([{"a": 1}], RESET) == []
    assert add_findings([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------- regression: Postgres lock
def test_stuck_scan_does_not_deadlock_on_a_locking_checkpointer(make_client, settings):
    """AsyncPostgresSaver guards its connection with a lock that an open alist()
    keeps holding. The first version of stuck_threads() called aget_state() inside
    the alist() loop and froze forever on the real stack. This saver reproduces
    that lock so the test fails (times out) if the pattern ever comes back."""
    import asyncio

    import fakeredis
    from fastapi.testclient import TestClient
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    from app.knowledge_client import KnowledgeClient
    from app.main import create_app
    from app.persistence import Persistence

    from conftest import FakeHub, knowledge_transport

    class LockingSaver(InMemorySaver):
        def __init__(self):
            super().__init__()
            self._lock = None

        def _get_lock(self):
            if self._lock is None:
                self._lock = asyncio.Lock()
            return self._lock

        async def alist(self, *a, **k):
            async with self._get_lock():            # held while the caller iterates
                async for item in super().alist(*a, **k):
                    yield item

        async def aget_tuple(self, config):
            async with self._get_lock():
                return await super().aget_tuple(config)

    async def ok():
        return True

    hub = FakeHub({"supervisor": [R("writer")], "writer": ["hi"]})
    kc = KnowledgeClient("http://knowledge", 5, transport=knowledge_transport()[0])
    app = create_app(settings.model_copy(update={"stuck_scan_interval_s": 0}), models=hub,
                     redis_client=fakeredis.FakeAsyncRedis(decode_responses=True), knowledge=kc,
                     persistence=Persistence(LockingSaver(), InMemoryStore(), "memory", ok))
    with TestClient(app) as c:
        c.post("/v1/chat", json={"message": "hi"})
        runner = app.state.runner
        # the scan must finish quickly, not hang on the lock
        result = c.portal.call(lambda: asyncio.wait_for(runner.stuck_threads(0), timeout=5))
    assert result == {"pending_approval": [], "abandoned": []}
