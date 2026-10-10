"""Helpers shared by the nodes: progress events, safe structured decisions,
prompt formatting and per-node tracing/metrics."""
import functools
import logging
import time

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, ValidationError

from .. import metrics as m
from ..events import StepEvent
from ..logging_setup import log_extra
from ..tracing import span

log = logging.getLogger("agent.graph")


def emit(agent: str, action: str, detail: str = "", status: str = "ok", **extra) -> None:
    """Send a progress event to the client (stream_mode="custom").
    The runner forwards it to the browser as an SSE `step` event."""
    try:
        writer = get_stream_writer()
    except RuntimeError:            # called outside a graph run (unit tests of a helper)
        return
    # validated HERE, so a bad event fails in the node that sent it, not in the browser
    writer(StepEvent(agent=agent, action=action, detail=detail, status=status, **extra).model_dump())


def _error_text(exc: ValidationError) -> str:
    """Pydantic errors in a form the model can act on: 'relevant: passage numbers must be...'"""
    parts = []
    for e in exc.errors(include_url=False):
        loc = ".".join(str(x) for x in e["loc"])
        parts.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return "; ".join(parts)


async def decide(runnable, messages: list, schema: type[BaseModel], fallback, *,
                 context: dict | None = None, retries: int = 1):
    """Run a structured-output call and VALIDATE it with Pydantic, self-correcting.

    `runnable` is model.with_structured_output(schema, include_raw=True): we get
    the model's raw tool call and validate its arguments ourselves, because
    LangChain's parser does not know our run-time `context`.

      valid                  -> return it
      invalid, retries left  -> send the errors back as the tool result, ask again
      still invalid          -> validate in "lenient" mode (validators repair)
      unusable / no call     -> `fallback`
    Network/auth errors are NOT swallowed: they propagate to the runner."""
    name = schema.__name__
    msgs, last_args = list(messages), None
    for attempt in range(retries + 1):
        out = await runnable.ainvoke(msgs)
        raw = out.get("raw") if isinstance(out, dict) else None
        calls = list(getattr(raw, "tool_calls", None) or [])
        if not calls:
            break
        last_args = calls[0]["args"]
        try:
            result = schema.model_validate(last_args, context=context)
        except ValidationError as exc:
            errors = _error_text(exc)
            if (getattr(raw, "response_metadata", None) or {}).get("finish_reason") == "length":
                # missing fields at the end = the reply was cut off, not a wrong answer
                errors = ("your reply was CUT OFF at the output token limit, so it is incomplete. "
                          "Be much more concise (fewer, shorter items). Details: " + errors)
            log.warning("structured output rejected", extra=log_extra(schema=name, attempt=attempt + 1,
                                                                      errors=errors[:300]))
            if attempt < retries:
                # OpenAI protocol: every tool call needs a tool message before the next turn
                msgs += [raw, *(ToolMessage(
                    f"Rejected: {errors}\nCall {c['name']} again with corrected values." if i == 0 else "ignored",
                    tool_call_id=c.get("id") or f"call_{i}", status="error") for i, c in enumerate(calls))]
            continue
        m.STRUCTURED_OUTPUTS.labels(name, "valid" if attempt == 0 else "corrected").inc()
        return result
    if last_args is not None:
        try:
            result = schema.model_validate(last_args, context={**(context or {}), "lenient": True})
            m.STRUCTURED_OUTPUTS.labels(name, "repaired").inc()
            return result
        except ValidationError:
            pass
    m.STRUCTURED_OUTPUTS.labels(name, "fallback").inc()
    return fallback


def to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):          # content blocks: [{"type": "text", "text": ...}, ...]
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def history(state: dict, limit: int) -> list[AnyMessage]:
    """Earlier turns of this conversation (the last message is the current question)."""
    msgs = [x for x in (state.get("messages") or [])[:-1] if isinstance(x, (HumanMessage, AIMessage))]
    return msgs[-limit:] if limit else []


MIN_NEW_CHARS = 80      # a neighbour chunk that adds less than this is dropped


def _overlap(left: str, right: str) -> int:
    """Length of the longest suffix of `left` that is also a prefix of `right`.
    Neighbouring chunks share text (chunk_overlap), so this finds the duplicate."""
    for k in range(min(len(left), len(right)), 0, -1):
        if left.endswith(right[:k]):
            return k
    return 0


def _new_text(p: dict, kept: dict[tuple[str, int], str]) -> str:
    """The passage text minus what an already-kept neighbour chunk repeats."""
    text = p["text"]
    prev = kept.get((p["doc_id"], p["chunk_index"] - 1))
    if prev:
        text = text[_overlap(prev, text):]
    nxt = kept.get((p["doc_id"], p["chunk_index"] + 1))
    if nxt:
        cut = _overlap(text, nxt)
        text = text[:len(text) - cut] if cut else text
    return text.strip()


def numbered_sources(sources: dict | None, max_n: int, max_chars: int) -> list[dict]:
    """Best passages first, numbered S1..Sn, total text bounded by max_chars.
    Text that a better-ranked neighbour chunk already shows is removed, so the
    writer never sees (and cites) the same sentences twice.
    Deterministic, so the writer, the grounding check and the final response
    all agree on which passage is S1."""
    ranked = sorted((sources or {}).values(), key=lambda p: (-p["score"], p["key"]))
    out, used, kept = [], 0, {}
    for p in ranked:
        if len(out) >= max_n:
            break
        text = _new_text(p, kept)
        if len(text) < MIN_NEW_CHARS and len(text) < len(p["text"]):
            continue                                  # only repeated its neighbour
        kept[(p["doc_id"], p["chunk_index"])] = p["text"]
        i = len(out) + 1
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        used += len(text)
        out.append({"source_id": f"S{i}", "doc_id": p["doc_id"], "filename": p["filename"],
                    "page": p["page"], "score": p["score"], "text": text})
        if used >= max_chars:
            break
    return out


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "(no passages)"
    return "\n\n".join(f"[{s['source_id']}] {s['filename']}, page {s['page']}:\n{s['text']}" for s in sources)


def format_findings(findings: list[dict] | None) -> str:
    if not findings:
        return "(none yet)"
    return "\n".join(f"- [{f['agent']}] {f['task']} -> {f['result']}" for f in findings)


def node(name: str):
    """Decorator for graph nodes: an OpenTelemetry span + Prometheus metrics.
    An interrupt is not an error: LangGraph raises GraphInterrupt to pause."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state, runtime):
            start, status = time.perf_counter(), "ok"
            with span(f"agent_node {name}", **{"gen_ai.agent.name": name}):
                try:
                    return await fn(state, runtime)
                except GraphInterrupt:
                    status = "interrupted"
                    raise
                except Exception:
                    status = "error"
                    raise
                finally:
                    m.NODE_RUNS.labels(name, status).inc()
                    m.NODE_LATENCY.labels(name).observe(time.perf_counter() - start)
        return wrapper
    return deco
