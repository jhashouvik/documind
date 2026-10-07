"""The agent loop (a ReAct-style tool-calling loop).

    messages = system prompt + conversation history + new question
    repeat up to max_steps times:
        ask the LLM (streaming), offering the tool schemas
        if it answered with text        -> that is the final answer, stop
        if it asked for tool calls      -> run them (in parallel), append the
                                           results as `tool` messages, loop again
    on the last step tools are withheld, forcing a text answer

The loop yields AgentEvents so the HTTP layer can stream progress to the
browser (Server-Sent Events) or collect them into one JSON response.

Tracing: the whole turn is one `invoke_agent` span; every LLM round trip is a
`chat <model>` span and every tool an `execute_tool <name>` span, using the
OpenTelemetry GenAI semantic conventions (gen_ai.*). In Jaeger you see the
turn as a tree, including the HTTP calls to OpenRouter and knowledge-service.
"""
import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

from . import metrics as m
from .llm import LLMClient, LLMError, parse_arguments
from .logging_setup import log_extra
from .memory import ConversationMemory
from .options import RunConfig
from .prompts import system_prompt
from .tools import CitationRegistry, ToolContext, execute_tool, to_tool_message, tool_schemas
from .tracing import span

log = logging.getLogger(__name__)
CITE_RE = re.compile(r"\[S(\d+)\]")


@dataclass
class AgentEvent:
    type: str            # session | step | token | done | error
    data: dict

    def sse(self) -> str:
        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


class Agent:
    def __init__(self, llm: LLMClient, memory: ConversationMemory, knowledge, settings) -> None:
        self.llm = llm
        self.memory = memory
        self.knowledge = knowledge
        self.s = settings

    async def run(self, session_id: str, question: str, cfg: RunConfig) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        yield AgentEvent("session", {"session_id": session_id, "version": self.s.app_version,
                                     "track": self.s.track, "model": cfg.llm_model})
        with span("invoke_agent documind", **{"gen_ai.operation.name": "invoke_agent",
                                              "gen_ai.agent.name": "documind",
                                              "gen_ai.conversation.id": session_id,
                                              "gen_ai.request.model": cfg.llm_model,
                                              "documind.prompt_version": cfg.prompt_version,
                                              "documind.track": self.s.track}) as turn:
            if self.s.trace_content:
                turn.add_event("gen_ai.user.message", {"content": question[:4000]})
            async for ev in self._loop(session_id, question, cfg, started, turn):
                yield ev

    async def _loop(self, session_id, question, cfg: RunConfig, started, turn) -> AsyncIterator[AgentEvent]:
        history = await self.memory.load(session_id)
        messages: list[dict] = [{"role": "system", "content": system_prompt(cfg.prompt_version)},
                                *history, {"role": "user", "content": question}]
        citations = CitationRegistry()
        ctx = ToolContext(self.knowledge, citations, cfg.top_k, self.s.tool_result_max_chars,
                          cfg.score_threshold, cfg.embedding_model)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        answer, model_used, steps = "", cfg.llm_model, 0
        model = cfg.llm_model

        try:
            for step in range(1, cfg.max_steps + 1):
                steps = step
                last_step = step == cfg.max_steps
                text_parts: list[str] = []
                end = None
                t0 = time.perf_counter()
                first_token_at = None

                with span(f"chat {model}", **{"gen_ai.operation.name": "chat", "gen_ai.system": "openrouter",
                                              "gen_ai.request.model": model,
                                              "gen_ai.request.temperature": cfg.temperature,
                                              "gen_ai.request.max_tokens": cfg.max_tokens,
                                              "documind.agent.step": step,
                                              "documind.tools_offered": not last_step}) as s:
                    async for ev in self.llm.stream_chat(messages, None if last_step else tool_schemas(),
                                                         model=model, temperature=cfg.temperature,
                                                         max_tokens=cfg.max_tokens):
                        if ev.kind == "content":
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                                m.LLM_TTFT.labels(model).observe(first_token_at - t0)
                            text_parts.append(ev.text)
                            yield AgentEvent("token", {"text": ev.text, "step": step})
                        else:
                            end = ev
                    if end:
                        s.set_attribute("gen_ai.response.model", end.model or model)
                        s.set_attribute("gen_ai.response.finish_reasons", [end.finish_reason or "unknown"])
                        s.set_attribute("documind.tool_calls", len(end.tool_calls))
                        if end.usage:
                            s.set_attribute("gen_ai.usage.input_tokens", end.usage.get("prompt_tokens") or 0)
                            s.set_attribute("gen_ai.usage.output_tokens",
                                            end.usage.get("completion_tokens") or 0)

                m.LLM_REQUESTS.labels(model, "ok").inc()
                m.LLM_LATENCY.labels(model).observe(time.perf_counter() - t0)
                model_used = (end.model if end else None) or model_used
                if end and end.usage:
                    for k in usage:
                        usage[k] += end.usage.get(k) or 0

                calls = end.tool_calls if end else []
                if not calls:
                    answer = "".join(text_parts).strip()
                    break

                # ---- the model wants to act: run every requested tool -------------
                messages.append({"role": "assistant", "content": "".join(text_parts) or None,
                                 "tool_calls": [c.as_message() for c in calls]})
                for c in calls:
                    yield AgentEvent("step", {"step": step, "tool": c.name, "status": "started",
                                              "arguments": _safe_json(c.arguments)})
                results = await asyncio.gather(*(self._run_call(c, ctx) for c in calls))
                for c, (result, status, ms) in zip(calls, results):
                    messages.append({"role": "tool", "tool_call_id": c.id, "name": c.name,
                                     "content": to_tool_message(result)})
                    yield AgentEvent("step", {"step": step, "tool": c.name, "status": status,
                                              "ms": ms, "summary": _summarise(c.name, result)})

            if not answer:
                answer = "I could not produce an answer within the allowed number of steps."
        except LLMError as exc:
            m.LLM_REQUESTS.labels(model, "error").inc()
            m.CHAT_REQUESTS.labels("llm_error").inc()
            log.error("llm error", extra=log_extra(error=str(exc), status=exc.status, model=model))
            yield AgentEvent("error", {"message": str(exc), "status": exc.status})
            return

        # ---- bookkeeping ------------------------------------------------------------
        for k, v in usage.items():
            m.LLM_TOKENS.labels(model, k.replace("_tokens", "")).inc(v)
        m.AGENT_STEPS.observe(steps)
        m.CHAT_REQUESTS.labels("ok").inc()
        await self.memory.append(session_id, {"role": "user", "content": question},
                                 {"role": "assistant", "content": answer})

        cited_ids = {f"S{n}" for n in CITE_RE.findall(answer)}
        sources = [asdict(s) for s in ctx.citations.sources.values()]
        took = int((time.perf_counter() - started) * 1000)
        turn.set_attribute("documind.steps", steps)
        turn.set_attribute("gen_ai.usage.input_tokens", usage["prompt_tokens"])
        turn.set_attribute("gen_ai.usage.output_tokens", usage["completion_tokens"])
        turn.set_attribute("documind.sources", len(sources))
        turn.set_attribute("documind.cited", len(cited_ids))
        if self.s.trace_content:
            turn.add_event("gen_ai.assistant.message", {"content": answer[:4000]})
        log.info("chat turn complete", extra=log_extra(
            session_id=session_id, steps=steps, sources=len(sources), cited=len(cited_ids), model=model,
            prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"], ms=took))
        yield AgentEvent("done", {
            "session_id": session_id, "answer": answer,
            "citations": [s for s in sources if s["source_id"] in cited_ids],
            "sources": sources, "steps": steps, "usage": usage, "model": model_used,
            "version": self.s.app_version, "track": self.s.track, "took_ms": took,
            "settings": cfg.as_dict()})

    async def _run_call(self, call, ctx: ToolContext) -> tuple[dict, str, int]:
        t0 = time.perf_counter()
        with span(f"execute_tool {call.name}", **{"gen_ai.operation.name": "execute_tool",
                                                  "gen_ai.tool.name": call.name,
                                                  "gen_ai.tool.call.id": call.id}) as s:
            try:
                args = parse_arguments(call.arguments)
            except (json.JSONDecodeError, ValueError) as exc:
                s.set_attribute("documind.tool.status", "invalid")
                return {"error": f"Arguments were not valid JSON ({exc}). Call the tool again "
                                 f"with a JSON object."}, "invalid", 0
            result, status = await execute_tool(call.name, args, ctx)
            s.set_attribute("documind.tool.status", status)
            return result, status, int((time.perf_counter() - t0) * 1000)


def _safe_json(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _summarise(tool: str, result: dict) -> str:
    if "error" in result:
        return result["error"][:160]
    if tool == "search_knowledge_base":
        files = sorted({r["filename"] for r in result.get("results", [])})
        return f"{len(result.get('results', []))} passages from {', '.join(files) or 'no documents'}"
    if tool == "list_documents":
        return f"{len(result.get('documents', []))} documents"
    if tool == "calculator":
        return f"{result.get('expression')} = {result.get('result')}"
    return "done"
