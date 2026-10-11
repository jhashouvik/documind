"""The contracts of agent-to-agent communication, and how a broken handoff is reported.

Agents never call each other: they write fields of the shared state that the
next agent reads. Every such handoff can lose information WITHOUT any error:

    supervisor -> worker      `instruction`   (empty? a repeat of an earlier task?)
    worker -> supervisor      `findings`      (search failed? nothing relevant? report cut short?)
    researcher -> writer      `sources`       (more passages than the writer's budget?)
    supervisor -> writer      `writer_brief`  (e.g. "give the hours too")
    grader -> researcher      a verdict       (unusable -> "keep everything")

So each handoff is TYPED (Finding, HandoffIssue: Pydantic models, validated where
they are created) and every degradation becomes a HandoffIssue that is:
  * added to the state (`issues`) and returned with the answer,
  * shown live in the trace (a `step` event from "handoff"),
  * counted in Prometheus: documind_handoff_issues_total{source, target, kind}.
GET /v1/sessions/{id}/handoffs replays which agent passed what to whom, step by step.
"""
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from .. import metrics as m
from .common import emit, to_text

Status = Literal["ok", "empty", "partial", "error"]
IssueKind = Literal[
    "search_failed",       # researcher -> supervisor: the knowledge base did not answer
    "no_evidence",         # researcher -> supervisor: searched, nothing relevant
    "grader_fallback",     # grader -> researcher: verdict unusable, all passages kept unchecked
    "truncated",           # researcher -> writer: passages dropped/cut by the context budget
    "partial_report",      # analyst/librarian -> supervisor: sub-agent stopped before finishing
    "empty_instruction",   # supervisor -> worker: no task given, worker falls back to the question
    "repeated_task",       # supervisor -> worker: same task again, rerouted to the writer
    "step_limit",          # supervisor -> writer: step budget used up, forced to answer
    "fallback_decision",   # supervisor: no valid routing decision, defaulted to the writer
    "check_unverified",    # grounding -> finalize: the fact check produced no verdict
    # ---- MISSED handoffs: something that should have been passed on never was ----
    "no_research",         # supervisor -> writer: a document question answered without any search
    "empty_answer",        # writer -> you: the writer produced nothing
    "abandoned_run",       # previous turn stopped halfway (pod killed, timeout): its last steps never ran
    "approval_abandoned",  # a paused approval was dropped because a new message arrived instead
]
LIMIT_PREFIX = "Model call limits exceeded"      # what ModelCallLimitMiddleware writes on exit


class Finding(BaseModel):
    agent: str = Field(min_length=1)
    task: str
    result: str = Field(min_length=1)
    status: Status = "ok"
    data: dict[str, Any] = Field(default_factory=dict)


class HandoffIssue(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: IssueKind
    detail: str = ""


def finding(agent: str, task: str, result: str, status: Status = "ok", **data) -> dict:
    """Validated at creation; stored as a plain dict (checkpoints stay version-tolerant)."""
    return Finding(agent=agent, task=task, result=result, status=status, data=data).model_dump()


def issue(source: str, target: str, kind: IssueKind, detail: str = "") -> dict:
    item = HandoffIssue(source=source, target=target, kind=kind, detail=detail)
    m.HANDOFF_ISSUES.labels(source, target, kind).inc()
    emit("handoff", f"{source} -> {target}", f"{kind}: {detail}" if detail else kind,
         "error" if kind in ("search_failed", "partial_report", "check_unverified", "no_research",
                             "empty_answer", "abandoned_run", "approval_abandoned") else "info")
    return item.model_dump()


def sub_agent_report(messages: list) -> tuple[str, Status, dict]:
    """Read a create_agent sub-agent's result honestly.

    When ModelCallLimitMiddleware stops an agent it appends an AIMessage
    "Model call limits exceeded: ..." - taking messages[-1] as "the report"
    would hand THAT sentence to the supervisor as if it were a result.
    """
    tool_results = [to_text(x.content) for x in messages if isinstance(x, ToolMessage)]
    data = {"tool_calls": sum(len(x.tool_calls) for x in messages if isinstance(x, AIMessage)),
            "tool_errors": sum(1 for x in messages if isinstance(x, ToolMessage) and x.status == "error")}
    last = messages[-1] if messages else None
    text = to_text(last.content).strip() if last is not None else ""
    if isinstance(last, AIMessage) and text.startswith(LIMIT_PREFIX):
        data["stopped_by"] = text
        so_far = "; ".join(r[:200] for r in tool_results[-3:]) or "no tool results"
        return f"INCOMPLETE: stopped by its call limit. Last tool results: {so_far}", "partial", data
    if isinstance(last, AIMessage) and last.tool_calls:
        return "INCOMPLETE: ended while still calling tools", "partial", data
    if not text:
        return "INCOMPLETE: no report", "partial", data
    return text, "ok", data
