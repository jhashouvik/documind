"""Nodes of the top-level DocuMind graph.

A node is `async def node(state, runtime) -> dict | Command`.
  * returning a dict      = "merge this into the state", then follow the edges
  * returning a Command   = merge `update` AND jump to `goto` (dynamic routing)
  * Command(goto=[Send(node, payload), ...]) = run that node once per payload,
    IN PARALLEL (map-reduce); the reducers in state.py merge the results.

Flow of one turn:

  start_turn -> supervisor <-------------------------------+
                  | Command(goto=...)                      |
                  +-> researcher (corrective-RAG subgraph) -+
                  +-> planner --Send x N--> research_task --+
                  +-> analyst   (create_agent + calculator) +
                  +-> librarian (create_agent + HITL)  -----+
                  +-> writer -> grounding_check --(unsupported claims)--> writer
                                       +--> finalize -> remember -> END
"""
import re
import uuid
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from .. import metrics as m
from .common import (decide, emit, format_evidence, format_findings, format_sources, history, node,
                     numbered_sources, select_sources, to_text)
from .context import AgentContext
from .decisions import similar_task
from .handoffs import finding, issue, sub_agent_report
from .prompts import GROUNDING, MEMORY, PLANNER, supervisor_prompt, writer_prompt
from .research import RESEARCH_GRAPH
from .state import RESET, DocuMindState, Grounding, Memories, Plan, RouteDecision
from .workers import analyst_agent, librarian_agent

WORKERS = ("researcher", "planner", "analyst", "librarian", "writer")
CITE = re.compile(r"\[S\d+\]")
# "the documents do not cover X" needs no citation
NOT_COVERED = re.compile(r"\b(documents?|sources?|passages?)\b[^.]{0,60}\b(do not|does not|don't|"
                         r"doesn't|not)\b|\b(could not|cannot|can't|unable to) (find|confirm)", re.IGNORECASE)
ABOUT_ME = re.compile(r"\b(i|i'm|im|i am|my|me|we|our|call me)\b", re.IGNORECASE)


# fallbacks returned by decide() when the model gave no valid output (compared by identity)
UNVERIFIED = Grounding(grounded=True)
FALLBACK_ROUTE = RouteDecision(next="writer", reason="no valid decision")
EMPTY_ANSWER = ("I could not produce an answer to this question. Please try again or rephrase it.")


class ResearchTask(TypedDict):
    question: str


def memory_namespace(user_id: str) -> tuple[str, ...]:
    return ("documind", "users", user_id, "memories")


def _sources(state, ctx: AgentContext) -> list[dict]:
    s = ctx.settings
    return numbered_sources(state.get("sources"), s.writer_max_sources, s.tool_result_max_chars)


# ---------------------------------------------------------------------------- turn start
@node("start_turn")
async def start_turn(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    """Reset the per-turn fields and load long-term memories (LangGraph Store)."""
    memories: list[str] = []
    uid = state.get("user_id")
    if uid and runtime.store and runtime.context.settings.long_term_memory:
        items = await runtime.store.asearch(memory_namespace(uid), limit=20)
        memories = [i.value["fact"] for i in items if "fact" in i.value]
        if memories:
            emit("memory", "recall", f"{len(memories)} facts about you")
    # what the previous turn left unfinished (detected by the runner) opens this turn's issues
    carried = [issue(p["source"], p["target"], p["kind"], p["detail"]) for p in state.get("pending_issues") or []]
    return {"memories": memories, "sources": RESET, "findings": RESET, "issues": [RESET, *carried],
            "pending_issues": [], "plan": [],
            "instruction": "", "writer_brief": "", "steps": 0, "draft": 0, "answer": "", "feedback": "",
            "grounded": None}


# ---------------------------------------------------------------------------- supervisor
@node("supervisor")
async def supervisor(state: DocuMindState, runtime: Runtime[AgentContext]
                     ) -> Command[Literal["researcher", "planner", "analyst", "librarian", "writer"]]:
    ctx = runtime.context
    steps = state.get("steps") or 0
    if steps >= ctx.cfg.max_steps:
        emit("supervisor", "route", "step limit reached -> writer", "info")
        m.ROUTES.labels("writer").inc()
        return Command(goto="writer", update={"writer_brief": "", "issues": [issue(
            "supervisor", "writer", "step_limit",
            f"{steps} of {ctx.cfg.max_steps} steps used: answering with what was found so far")]})

    sources = state.get("sources") or {}
    findings = state.get("findings") or []
    research_done = any(f["agent"] in ("researcher", "planner") for f in findings)
    # the supervisor sees EXCERPTS of the evidence, not just "3 relevant passages":
    # otherwise it decides "enough information" without knowing what the passages say
    status = (f"Current question: {state['question']}\n\n"
              f"Findings so far:\n{format_findings(findings)}\n\n"
              f"Evidence collected ({len(sources)} relevant passages, best first):\n"
              f"{format_evidence(sources)}\n\n"
              f"Steps used: {steps} of {ctx.cfg.max_steps}. Choose the next worker. Choose 'writer' "
              f"only if the evidence above actually answers the question (or nothing more can be found).")
    decision = await decide(
        ctx.models.structured(ctx.cfg, "supervisor", RouteDecision),
        [SystemMessage(supervisor_prompt(state.get("memories") or [])),
         *history(state, ctx.settings.history_max_messages), HumanMessage(status)],
        RouteDecision, FALLBACK_ROUTE,
        # the validator rejects a repeated task (and tells the model what that task returned)
        # and a document question that tries to skip the research
        context={"done": {(f["agent"], f["task"].strip().lower()): f["result"] for f in findings},
                 "research_done": research_done})

    issues = []
    if decision is FALLBACK_ROUTE:
        issues.append(issue("supervisor", "writer", "fallback_decision",
                            "no valid routing decision from the model: answering now"))
    nxt = decision.next if decision.next in WORKERS else "writer"
    if nxt != "writer" and not decision.instruction:      # only possible after a lenient repair
        issues.append(issue("supervisor", nxt, "empty_instruction",
                            "no task given: the worker gets the user's whole question"))
    instruction = (decision.instruction or state["question"]).strip()
    if nxt != "writer" and any(f["agent"] == nxt and similar_task(f["task"], instruction)
                               for f in findings):        # loop guard (safety net after the validator)
        issues.append(issue("supervisor", nxt, "repeated_task",
                            f"'{instruction[:80]}' was already done: answering now"))
        nxt, decision.reason = "writer", "same task already done"
    if nxt == "writer" and decision.needs_documents and not research_done:
        # still skipping the research after the validator's correction: a MISSED handoff
        issues.append(issue("supervisor", "writer", "no_research",
                            "a question that needs the documents is answered without any search"))
    # an instruction meant for the writer travels in its own field: a stale worker
    # instruction must never be read as the writer's brief
    brief = decision.instruction if nxt == "writer" and decision.next == "writer" else ""
    m.ROUTES.labels(nxt).inc()
    emit("supervisor", "route", f"-> {nxt}: {instruction if nxt != 'writer' else (brief or decision.reason)}")
    update = {"steps": steps + 1, "writer_brief": brief, "issues": issues}
    if nxt != "writer":
        update["instruction"] = instruction
    return Command(goto=nxt, update=update)


# ---------------------------------------------------------------------------- research
async def _research(question: str, runtime: Runtime[AgentContext]) -> dict:
    out = await RESEARCH_GRAPH.ainvoke({"question": question}, context=runtime.context)
    relevant = out.get("relevant") or []
    tried = out.get("tried") or []
    issues = list(out.get("issues") or [])                 # e.g. grader_fallback from the subgraph
    if out.get("error"):
        report = finding("researcher", question, f"search failed: {out['error']}", "error", queries=tried)
        issues.append(issue("researcher", "supervisor", "search_failed", out["error"][:200]))
    elif relevant:
        files = sorted({p["filename"] for p in relevant})
        pages = sorted({p["page"] for p in relevant})
        report = finding("researcher", question,
                         f"{len(relevant)} relevant passages from {', '.join(files)} "
                         f"(pages {', '.join(map(str, pages))})", "ok",
                         passages=len(relevant), files=files, pages=pages, queries=tried)
    else:
        report = finding("researcher", question,
                         "no relevant passages found (queries tried: " + "; ".join(tried) + ")", "empty",
                         passages=0, queries=tried)
        issues.append(issue("researcher", "supervisor", "no_evidence", "queries tried: " + "; ".join(tried)))
    return {"sources": {p["key"]: p for p in relevant}, "findings": [report], "issues": issues}


@node("researcher")
async def researcher(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    return await _research(state.get("instruction") or state["question"], runtime)


@node("research_task")
async def research_task(task: ResearchTask, runtime: Runtime[AgentContext]) -> dict:
    """One parallel branch created by the planner's Send()."""
    return await _research(task["question"], runtime)


# ---------------------------------------------------------------------------- planner
@node("planner")
async def planner(state: DocuMindState, runtime: Runtime[AgentContext]
                  ) -> Command[Literal["research_task", "supervisor"]]:
    ctx = runtime.context
    n = ctx.settings.planner_max_subquestions
    task = state.get("instruction") or state["question"]
    plan = await decide(ctx.models.structured(ctx.cfg, "planner", Plan),
                        [SystemMessage(PLANNER.format(n=n)), HumanMessage(f"Question: {task}")],
                        Plan, Plan.model_construct(sub_questions=[task]), context={"max_subquestions": n})
    subs = list(dict.fromkeys(q.strip() for q in plan.sub_questions if q.strip()))[:n] or [task]
    emit("planner", "plan", " | ".join(subs))
    return Command(
        goto=[Send("research_task", {"question": q}) for q in subs],      # fan-out
        update={"plan": subs,
                "findings": [finding("planner", task,
                                     f"split into {len(subs)} sub-questions researched in parallel",
                                     sub_questions=subs)]})


# ---------------------------------------------------------------------------- prebuilt agents
def _facts(state, ctx: AgentContext) -> str:
    return (f"Sources:\n{format_sources(_sources(state, ctx))}\n\n"
            f"Findings:\n{format_findings(state.get('findings'))}")


@node("analyst")
async def analyst(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    task = state.get("instruction") or state["question"]
    agent = analyst_agent(ctx.models.chat(ctx.cfg, "analyst"))
    out = await agent.ainvoke({"messages": [HumanMessage(f"Instruction: {task}\n\n{_facts(state, ctx)}")]},
                              context=ctx)
    return _sub_agent_result("analyst", task, out["messages"])


def _sub_agent_result(agent: str, task: str, messages: list) -> dict:
    """The sub-agent's report, checked: a run stopped by its call limit is marked
    "partial" instead of passing the limit message on as if it were the answer."""
    report, status, data = sub_agent_report(messages)
    issues = [issue(agent, "supervisor", "partial_report", report[:200])] if status == "partial" else []
    return {"findings": [finding(agent, task, report, status, **data)], "issues": issues}


@node("librarian")
async def librarian(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    """May PAUSE inside (human approval of a deletion). On resume LangGraph
    re-runs this node from the top and the sub-agent continues from its
    checkpoint, so everything before ainvoke() must be safe to repeat."""
    ctx = runtime.context
    task = state.get("instruction") or state["question"]
    agent = librarian_agent(ctx.models.chat(ctx.cfg, "librarian"))
    out = await agent.ainvoke({"messages": [HumanMessage(
        f"User request: {state['question']}\nSupervisor instruction: {task}")]}, context=ctx)
    return _sub_agent_result("librarian", task, out["messages"])


# ---------------------------------------------------------------------------- writer + self-check
@node("writer")
async def writer(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    """Streams the answer. The runner forwards this node's tokens to the browser."""
    ctx = runtime.context
    draft = (state.get("draft") or 0) + 1
    feedback = state.get("feedback") or ""
    emit("writer", "draft", f"draft {draft}" + (" (revising unsupported claims)" if feedback else ""),
         draft=draft)
    s = ctx.settings
    selection = select_sources(state.get("sources"), s.writer_max_sources, s.tool_result_max_chars)
    issues = []
    if selection.lossy and draft == 1:     # researcher -> writer handoff lost passages: say so
        issues.append(issue("researcher", "writer", "truncated", f"writer {selection.describe()}"))
    body = (f"Question: {state['question']}\n\nNumbered sources:\n{format_sources(selection.numbered)}\n\n"
            f"Worker findings:\n{format_findings(state.get('findings'))}")
    if state.get("writer_brief"):
        body += f"\n\nInstruction from the supervisor for this answer: {state['writer_brief']}"
    if feedback:
        body += f"\n\nA reviewer rejected your previous draft:\n{feedback}\nRewrite the answer to fix this."
    resp = await ctx.models.chat(ctx.cfg, "writer").ainvoke(
        [SystemMessage(writer_prompt(ctx.cfg.prompt_version, state.get("memories") or [])),
         *history(state, ctx.settings.history_max_messages), HumanMessage(body)])
    return {"answer": to_text(resp.content).strip(), "draft": draft, "issues": issues}


@node("grounding_check")
async def grounding_check(state: DocuMindState, runtime: Runtime[AgentContext]
                          ) -> Command[Literal["writer", "finalize"]]:
    """Self-RAG: verify the draft against its sources; send it back once if needed."""
    ctx = runtime.context
    answer = state.get("answer") or ""
    draft = state.get("draft") or 1
    if not answer.strip():                 # the writer handed over NOTHING (a missed handoff)
        if draft <= ctx.settings.max_revisions:
            return Command(goto="writer", update={
                "feedback": "- Your previous draft was EMPTY. Write a complete answer.",
                "issues": [issue("writer", "grounding_check", "empty_answer",
                                 f"draft {draft} was empty: asking the writer again")]})
        return Command(goto="finalize", update={
            "answer": EMPTY_ANSWER, "grounded": None,
            "issues": [issue("writer", "finalize", "empty_answer",
                             f"still empty after {draft} drafts: sent a fallback message instead")]})
    sources = _sources(state, ctx)
    if not ctx.settings.grounding_check or not sources:
        return Command(goto="finalize", update={"grounded": None})
    verdict = await decide(
        ctx.models.structured(ctx.cfg, "grounding", Grounding),
        [SystemMessage(GROUNDING),
         HumanMessage(f"Sources:\n{format_sources(sources)}\n\nWorker findings:\n"
                      f"{format_findings(state.get('findings'))}\n\nAnswer to check:\n{answer}")],
        Grounding, UNVERIFIED,                                       # sentinel: the check did not run
        # the validator checks that every quote is really in the source it cites
        context={"sources": {**{s["source_id"]: s["text"] for s in sources},
                             "findings": format_findings(state.get("findings"))}})
    unsupported = verdict.problems()
    uncited = uncited_paragraphs(answer)                             # deterministic: no LLM needed
    problems = unsupported + [f'This paragraph has no citation, add [S#] to each sentence: "{p[:120]}"'
                              for p in uncited]
    grounded = verdict.grounded and not problems
    verified = verdict is not UNVERIFIED          # False: the checker never produced a valid verdict
    m.GROUNDING_CHECKS.labels("unverified" if not verified and grounded
                              else "grounded" if grounded else "ungrounded").inc()
    if grounded and not verified:                 # never report "fact-checked" for a check that did not run
        emit("grounding", "check", "fact check could not run (invalid output): answer NOT verified", "info")
        return Command(goto="finalize", update={"grounded": None, "issues": [issue(
            "grounding", "finalize", "check_unverified", "no valid verdict after the retry")]})
    if grounded or draft > ctx.settings.max_revisions:
        emit("grounding", "check", f"all {len(verdict.claims)} claims supported" if grounded
             else "still unsupported claims after revision", "ok" if grounded else "error")
        return Command(goto="finalize", update={"grounded": grounded})
    detail = ", ".join(x for x in (f"{len(unsupported)} unsupported claims" if unsupported else "",
                                   f"{len(uncited)} uncited paragraphs" if uncited else "") if x)
    emit("grounding", "check", f"{detail} -> revise", "error")
    return Command(goto="writer", update={"grounded": False,
                                          "feedback": "Claims NOT supported by the sources, or missing "
                                                      "citations:\n" + "\n".join(f"- {p}" for p in problems)})


BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _units(answer: str) -> list[str]:
    """Paragraphs, but every bullet of a list counts on its own (with its
    continuation lines), so one citation at the end of a list is not enough.
    An indented block after a blank line (e.g. a formula) continues the bullet above."""
    units: list[str] = []
    for block in re.split(r"\n[ \t]*\n", answer):    # keep the indentation of the next block
        if not block.strip():
            continue
        if block[:1] in " \t" and units:
            units[-1] += "\n\n" + block.strip()
            continue
        items: list[str] = []
        for line in block.strip().splitlines():
            if BULLET.match(line) or not items:
                items.append(line)
            else:
                items[-1] += "\n" + line
        units.extend(i.strip() for i in items if i.strip())
    return units


def uncited_paragraphs(answer: str) -> list[str]:
    """Factual paragraphs/bullets without any [S#]. Skipped: short lines,
    lead-ins ending with ':' and "the documents do not cover this" statements."""
    return [t for t in _units(answer)
            if len(t) >= 40 and not t.endswith(":") and not CITE.search(t) and not NOT_COVERED.search(t)]


# ---------------------------------------------------------------------------- end of turn
@node("finalize")
async def finalize(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    """Only the final answer joins the conversation history (not the drafts,
    not the workers' chatter): history is re-sent every turn and tokens cost money."""
    return {"messages": [AIMessage(state.get("answer") or "")]}


@node("remember")
async def remember(state: DocuMindState, runtime: Runtime[AgentContext]) -> dict:
    """Long-term memory: durable facts about the user, stored in the LangGraph
    Store (Postgres) under the user's namespace. Unlike the checkpointer, the
    Store is shared across ALL sessions of that user."""
    ctx = runtime.context
    uid = state.get("user_id")
    question = state.get("question") or ""
    if not (uid and runtime.store and ctx.settings.long_term_memory and ABOUT_ME.search(question)):
        return {}                     # cheap gate: no LLM call unless the user talks about themselves
    extracted = await decide(ctx.models.structured(ctx.cfg, "memory", Memories),
                             [SystemMessage(MEMORY), HumanMessage(question)], Memories, Memories())
    known = {f.lower() for f in state.get("memories") or []}
    new = [f.strip() for f in extracted.facts if f.strip() and f.strip().lower() not in known]
    for fact in new:
        await runtime.store.aput(memory_namespace(uid), uuid.uuid4().hex, {"fact": fact})
    if new:
        emit("memory", "save", "; ".join(new))
    return {}
