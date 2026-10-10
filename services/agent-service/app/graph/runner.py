"""Run, resume and replay the graph; translate its stream into SSE events.

graph.astream(..., stream_mode=["updates", "messages", "custom"], subgraphs=True, version="v2")
yields dicts {"type": ..., "ns": (...), "data": ...}:

    custom    progress events our nodes/tools emit with get_stream_writer()  -> SSE `step`
    messages  (token chunk, metadata) for EVERY model call in the graph; we
              forward only the writer node's tokens                          -> SSE `token`
    updates   state changes per node; an interrupt shows up here as
              "__interrupt__"

ns ("namespace") says where an event came from: () = the top-level graph,
("librarian:<task id>",) = inside the librarian sub-agent, and so on.

SSE events sent to the browser:
    session    {session_id, version, track, model, engine}
    step       {agent, action, detail, status}
    token      {text, draft}             draft changes when the answer is rewritten
    interrupt  {session_id, actions[...], allowed[...]}  the graph is paused
    done       {answer, citations, sources, plan, findings, grounded, usage, ...}
    error      {message, status}
"""
import logging
import re
import time
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from .. import metrics as m
from ..events import (AgentEvent, DoneEvent, ErrorEvent, InterruptAction, InterruptEvent, SessionEvent,
                      StepEvent, TokenEvent)
from ..logging_setup import log_extra
from ..schemas import ResumeDecision
from ..models import UsageCallback, friendly_error
from ..options import RunConfig
from ..tracing import span
from .common import numbered_sources, to_text
from .context import AgentContext

log = logging.getLogger("agent.runner")
CITE_RE = re.compile(r"\[(S\d+)\]")
STREAM_MODES = ["updates", "messages", "custom"]


class NothingToResume(LookupError):
    pass


class AgentRunner:
    def __init__(self, graph, settings, models, knowledge) -> None:
        self.graph = graph
        self.s = settings
        self.models = models
        self.knowledge = knowledge

    @staticmethod
    def thread(session_id: str, checkpoint_id: str | None = None) -> dict:
        conf = {"thread_id": session_id}
        if checkpoint_id:
            conf["checkpoint_id"] = checkpoint_id
        return {"configurable": conf}

    # ------------------------------------------------------------------ entry points
    def run(self, session_id: str, question: str, cfg: RunConfig, user_id: str | None
            ) -> AsyncIterator[AgentEvent]:
        inp = {"messages": [HumanMessage(question)], "question": question,
               "run_config": cfg.as_dict(), "user_id": user_id}
        return self._drive(session_id, inp, cfg, user_id, "new")

    async def pending(self, session_id: str):
        """The paused state of a thread, or NothingToResume."""
        snap = await self.graph.aget_state(self.thread(session_id))
        if not snap.interrupts:
            raise NothingToResume(f"session {session_id} is not waiting for a decision")
        return snap

    @staticmethod
    def pending_actions(snap) -> list[dict]:
        """The tool calls waiting for a human: [{"name": ..., "args": ...}, ...]."""
        return [req for intr in snap.interrupts if isinstance(intr.value, dict)
                for req in intr.value.get("action_requests", [])]

    @staticmethod
    def allowed_decisions(snap) -> set[str]:
        return {d for intr in snap.interrupts if isinstance(intr.value, dict)
                for rc in intr.value.get("review_configs", []) for d in rc.get("allowed_decisions", [])}

    def resume(self, snap, session_id: str, decision: ResumeDecision) -> AsyncIterator[AgentEvent]:
        """Command(resume=...) is the value interrupt() returns inside the paused
        node. HumanInTheLoopMiddleware expects {"decisions": [one per action]}:
            {"type": "approve"}
            {"type": "reject", "message": "..."}
            {"type": "edit", "edited_action": {"name": tool, "args": {...}}}"""
        m.INTERRUPTS.labels(decision.decision).inc()

        def to_middleware(action: dict) -> dict:
            if decision.decision == "edit":
                return {"type": "edit", "edited_action": {"name": action["name"], "args": decision.args}}
            if decision.decision == "reject":
                return {"type": "reject",
                        "message": decision.message or "The user rejected this action. Do not retry it."}
            return {"type": "approve"}

        def answer(intr) -> dict:
            actions = intr.value.get("action_requests", []) or [{"name": ""}]
            return {"decisions": [to_middleware(a) for a in actions]}
        if len(snap.interrupts) == 1:
            cmd = Command(resume=answer(snap.interrupts[0]))
        else:
            cmd = Command(resume={i.id: answer(i) for i in snap.interrupts})
        values = snap.values
        return self._drive(session_id, cmd, RunConfig(**values["run_config"]), values.get("user_id"), "resume")

    async def replay(self, session_id: str, checkpoint_id: str) -> AsyncIterator[AgentEvent]:
        """Time travel: continue from an OLD checkpoint. LangGraph forks the
        thread there; the new branch becomes the thread's latest state."""
        snap = await self.graph.aget_state(self.thread(session_id, checkpoint_id))
        if not snap.values:
            raise NothingToResume(f"checkpoint {checkpoint_id} not found")
        values = snap.values
        cfg = RunConfig(**values["run_config"]) if values.get("run_config") else None
        if cfg is None:
            raise NothingToResume("that checkpoint is before the first question")
        return self._drive(session_id, None, cfg, values.get("user_id"), "replay", checkpoint_id)

    # ------------------------------------------------------------------ sessions
    async def messages(self, session_id: str) -> list[dict]:
        snap = await self.graph.aget_state(self.thread(session_id))
        out = []
        for msg in (snap.values or {}).get("messages", []):
            role = "user" if msg.type == "human" else "assistant"
            out.append({"role": role, "content": to_text(msg.content)})
        return out

    async def forget(self, session_id: str) -> None:
        await self.graph.checkpointer.adelete_thread(session_id)

    async def checkpoints(self, session_id: str, limit: int = 50) -> list[dict]:
        out = []
        async for snap in self.graph.aget_state_history(self.thread(session_id), limit=limit):
            meta = snap.metadata or {}
            out.append({"checkpoint_id": snap.config["configurable"]["checkpoint_id"],
                        "step": meta.get("step"), "source": meta.get("source"),
                        "next": list(snap.next), "created_at": snap.created_at,
                        "question": (snap.values or {}).get("question"),
                        "answer_preview": ((snap.values or {}).get("answer") or "")[:100],
                        "interrupted": bool(snap.interrupts)})
        return out

    # ------------------------------------------------------------------ the stream
    async def _drive(self, session_id, inp, cfg: RunConfig, user_id, mode: str,
                     checkpoint_id: str | None = None) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        usage = UsageCallback()
        ctx = AgentContext(cfg=cfg, settings=self.s, models=self.models, knowledge=self.knowledge,
                           user_id=user_id)
        config = {**self.thread(session_id, checkpoint_id), "callbacks": [usage],
                  "recursion_limit": self.s.graph_recursion_limit, "run_name": "documind",
                  "metadata": {"session_id": session_id, "track": self.s.track, "mode": mode}}
        yield SessionEvent(session_id=session_id, version=self.s.app_version, track=self.s.track,
                           model=cfg.llm_model, mode=mode)
        draft = 1
        with span("invoke_agent documind", **{"gen_ai.operation.name": "invoke_agent",
                                              "gen_ai.agent.name": "documind",
                                              "gen_ai.conversation.id": session_id,
                                              "gen_ai.request.model": cfg.llm_model,
                                              "documind.prompt_version": cfg.prompt_version,
                                              "documind.track": self.s.track,
                                              "documind.mode": mode}) as turn:
            try:
                # Consume the stream to the END (no return/raise inside the loop):
                # the turn span stays open across our yields.
                async for ev in self.graph.astream(inp, config, context=ctx, stream_mode=STREAM_MODES,
                                                   subgraphs=True, version="v2"):
                    if ev["type"] == "custom":
                        step = StepEvent.model_validate(ev["data"])     # nodes/tools emit StepEvent dumps
                        if step.action == "draft" and step.draft:
                            draft = step.draft
                        yield step
                    elif ev["type"] == "messages" and not ev["ns"]:
                        chunk, meta = ev["data"]
                        if meta.get("langgraph_node") == "writer" and isinstance(chunk, AIMessageChunk):
                            text = to_text(chunk.content)
                            if text:
                                yield TokenEvent(text=text, draft=draft)
            except GraphRecursionError:
                m.CHAT_REQUESTS.labels("recursion_limit").inc()
                yield ErrorEvent(message="The agents did not finish within the step budget.", status=500)
                return
            except Exception as exc:  # noqa: BLE001 - every failure becomes an error event
                err = friendly_error(exc)
                m.CHAT_REQUESTS.labels("llm_error" if err.status else "error").inc()
                log.error("agent run failed", extra=log_extra(error=str(err), status=err.status,
                                                              model=cfg.llm_model, mode=mode))
                yield ErrorEvent(message=str(err), status=err.status)
                return

            snap = await self.graph.aget_state(self.thread(session_id))
            turn.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            turn.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            if snap.interrupts:
                turn.set_attribute("documind.interrupted", True)
                m.CHAT_REQUESTS.labels("interrupted").inc()
                yield self._interrupt_event(session_id, snap.interrupts)
                return

            values = snap.values
            answer = values.get("answer") or ""
            sources = numbered_sources(values.get("sources"), self.s.writer_max_sources,
                                       self.s.tool_result_max_chars)
            cited = set(CITE_RE.findall(answer))
            findings = values.get("findings") or []
            took = int((time.perf_counter() - started) * 1000)
            steps = values.get("steps") or 0
            m.AGENT_STEPS.observe(steps)
            m.CHAT_REQUESTS.labels("ok").inc()
            turn.set_attribute("documind.steps", steps)
            turn.set_attribute("documind.sources", len(sources))
            turn.set_attribute("documind.cited", len(cited))
            if self.s.trace_content:
                turn.add_event("gen_ai.assistant.message", {"content": answer[:4000]})
            log.info("chat turn complete", extra=log_extra(
                session_id=session_id, mode=mode, steps=steps, sources=len(sources), cited=len(cited),
                agents=sorted({f["agent"] for f in findings}), grounded=values.get("grounded"),
                llm_calls=usage.calls, prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens, ms=took))
            yield DoneEvent(
                session_id=session_id, answer=answer,
                citations=[s for s in sources if s["source_id"] in cited], sources=sources,
                steps=steps, usage=usage.usage, llm_calls=usage.calls, model=cfg.llm_model,
                version=self.s.app_version, track=self.s.track, took_ms=took,
                settings=cfg.as_dict(), plan=values.get("plan") or [], findings=findings,
                grounded=values.get("grounded"), drafts=values.get("draft") or 0)

    @staticmethod
    def _interrupt_event(session_id: str, interrupts) -> InterruptEvent:
        actions, allowed = [], set()
        for intr in interrupts:
            value = intr.value if isinstance(intr.value, dict) else {"description": str(intr.value)}
            for req in value.get("action_requests", []):
                actions.append(InterruptAction(name=req.get("name") or "?", args=req.get("args") or {},
                                               description=req.get("description")))
            for rc in value.get("review_configs", []):
                allowed.update(rc.get("allowed_decisions", []))
        return InterruptEvent(session_id=session_id, actions=actions,
                              allowed=sorted(allowed & {"approve", "reject", "edit"}) or ["approve", "reject"])
