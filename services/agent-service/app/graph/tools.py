"""LangChain tools used by the prebuilt sub-agents (analyst, librarian).

`@tool` turns a Python function into a tool: the function name, the docstring
and the type hints become the JSON schema the LLM sees. The LLM decides
*whether* and *how* to call a tool; LangGraph's ToolNode actually runs it.

The special `runtime: ToolRuntime` parameter is NOT shown to the LLM. LangGraph
injects it, giving the tool the run context (knowledge client, settings), the
graph state and a stream writer for progress events.
"""
import ast
import json
import operator
import time
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from pydantic import Field

from .. import metrics as m
from ..events import StepEvent
from ..knowledge_client import KnowledgeUnavailable
from .context import AgentContext

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic WITHOUT eval(): walk the syntax tree and allow only
    numbers and arithmetic operators. eval() on model output would let a
    prompt-injected document run arbitrary Python inside your pod."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported element: {type(node).__name__}")
    result = ev(ast.parse(expression.replace(",", "").replace("^", "**"), mode="eval"))
    if abs(result) > 1e18:
        raise ValueError("result too large")
    return round(result, 10)


def _emit(runtime: ToolRuntime, agent: str, action: str, detail: str, status: str = "ok") -> None:
    if runtime.stream_writer:
        runtime.stream_writer(StepEvent(agent=agent, action=action, detail=detail, status=status).model_dump())


def _record(name: str, status: str, start: float) -> None:
    m.TOOL_CALLS.labels(name, status).inc()
    m.TOOL_LATENCY.labels(name).observe(time.perf_counter() - start)


# ---- argument types: Annotated[type, Field(...)] puts the limits into the JSON
# schema the LLM sees AND makes ToolNode reject a violating call before the tool
# runs (the model gets the Pydantic error back and can fix its call).
Expression = Annotated[str, Field(min_length=1, max_length=200, description=(
    "Numbers, + - * / // % ** and parentheses, e.g. '(250000000 * 0.15) / 12'."))]
DocId = Annotated[str, Field(pattern=r"^[0-9a-f]{16}$", description=(
    "The doc_id from list_documents: 16 lowercase hex characters."))]
Filename = Annotated[str, Field(min_length=1, max_length=255, description=(
    "The document's filename (shown to the human reviewer)."))]


@tool
def calculator(expression: Expression, runtime: ToolRuntime[AgentContext]) -> str:
    """Evaluate an arithmetic expression exactly."""
    start = time.perf_counter()
    try:
        result = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        _record("calculator", "error", start)
        _emit(runtime, "analyst", "calculator", f"{expression}: {exc}", "error")
        return json.dumps({"error": f"Cannot evaluate '{expression}': {exc}"})
    _record("calculator", "ok", start)
    _emit(runtime, "analyst", "calculator", f"{expression} = {result}")
    return json.dumps({"expression": expression, "result": result})


@tool
async def list_documents(runtime: ToolRuntime[AgentContext]) -> str:
    """List the documents currently in the knowledge base (doc_id, filename, pages, chunks)."""
    start = time.perf_counter()
    ctx = runtime.context
    try:
        docs = await ctx.knowledge.list_documents(ctx.cfg.embedding_model)
    except KnowledgeUnavailable as exc:
        _record("list_documents", "error", start)
        return json.dumps({"error": f"The knowledge base is temporarily unavailable: {exc}"})
    _record("list_documents", "ok", start)
    _emit(runtime, "librarian", "list_documents", f"{len(docs)} documents")
    return json.dumps({"documents": [{"doc_id": d.doc_id, "filename": d.filename, "pages": d.pages,
                                      "chunks": d.total_chunks} for d in docs]})


@tool
async def delete_document(doc_id: DocId, filename: Filename, runtime: ToolRuntime[AgentContext]) -> str:
    """Permanently delete a document and all its chunks from the knowledge base."""
    start = time.perf_counter()
    ctx = runtime.context
    try:
        resp = await ctx.knowledge.delete(doc_id, ctx.cfg.embedding_model)
    except KnowledgeUnavailable as exc:
        _record("delete_document", "error", start)
        return json.dumps({"error": f"The knowledge base is temporarily unavailable: {exc}"})
    if resp.status_code == 404:
        _record("delete_document", "error", start)
        return json.dumps({"error": f"No document with doc_id {doc_id}."})
    _record("delete_document", "ok", start)
    _emit(runtime, "librarian", "delete_document", f"deleted {filename}")
    return json.dumps({"deleted": doc_id, "filename": filename})


TOOLS = {t.name: t for t in (calculator, list_documents, delete_document)}
