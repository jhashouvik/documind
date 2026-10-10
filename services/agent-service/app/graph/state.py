"""Graph state, reducers and the schemas of every structured LLM decision.

STATE = one TypedDict shared by all nodes of the top-level graph. A node never
mutates it: it RETURNS a partial update ({"answer": "..."}) and LangGraph
merges it in. For most keys "merge" means overwrite. A key annotated with a
REDUCER is merged by that function instead, which is what makes parallel
branches safe: two research_task branches writing `sources` at the same time
are combined with merge_sources() instead of one overwriting the other.

The checkpointer saves this state after every super-step, keyed by thread_id
(= the chat session id). That is the agent's short-term memory: `messages`
survives across turns, pod restarts and replicas.
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

RESET = "__reset__"     # write this to a reducer key to empty it (start of a new turn)


class Passage(TypedDict):
    key: str            # "<doc_id>:<chunk_index>" - identity of a chunk
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    score: float
    text: str
    query: str          # the search query that found it


class Finding(TypedDict):
    agent: str          # researcher | analyst | librarian | planner
    task: str           # what the supervisor asked for
    result: str         # the worker's report back to the supervisor


def merge_sources(old: dict | None, new: dict | str | None) -> dict:
    """Union of passages by chunk key; keep the higher score on duplicates."""
    if new == RESET:
        return {}
    out = dict(old or {})
    for k, p in (new or {}).items():
        if k not in out or p["score"] > out[k]["score"]:
            out[k] = p
    return out


def add_findings(old: list | None, new: list | str | None) -> list:
    if new == RESET:
        return []
    return (old or []) + list(new or [])


class DocuMindState(TypedDict, total=False):
    # ---- persists across turns (conversation) ----
    messages: Annotated[list[AnyMessage], add_messages]   # user questions + final answers
    user_id: str | None
    # ---- reset at the start of every turn ----
    question: str
    run_config: dict            # RunConfig of this turn: needed again on resume / replay
    memories: list[str]         # long-term facts about this user (from the Store)
    sources: Annotated[dict[str, Passage], merge_sources]
    findings: Annotated[list[Finding], add_findings]
    plan: list[str]
    instruction: str            # supervisor -> worker
    steps: int                  # supervisor decisions so far
    draft: int                  # writer attempts so far
    answer: str
    feedback: str               # grounding check -> writer (what to fix)
    grounded: bool | None


class ResearchState(TypedDict, total=False):
    """State of the corrective-RAG subgraph (its own, smaller world)."""
    question: str
    query: str
    attempts: int
    tried: list[str]
    candidates: list[Passage]
    relevant: list[Passage]
    error: str


# The structured-output schemas live in decisions.py (with their validators);
# re-exported here so `from .state import Grades` keeps working.
from .decisions import (ClaimCheck, Grades, Grounding, Memories, Plan, RouteDecision,  # noqa: E402,F401
                        Rewrite, Worker)
