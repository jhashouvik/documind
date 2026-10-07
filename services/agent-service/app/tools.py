"""Tools the agent can call.

A tool = name + description + JSON-schema of its arguments + a Python
function. The LLM only ever sees the name, description and schema; it decides
*whether* and *how* to call a tool, and our code actually runs it.

Argument schemas are generated from Pydantic models, so the schema sent to
the model and the validation of what the model sends back can never drift
apart.
"""
import ast
import json
import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from . import metrics as m
from .knowledge_client import KnowledgeClient, KnowledgeUnavailable


# ---------------------------------------------------------------------------- citations
@dataclass
class Source:
    source_id: str
    doc_id: str
    filename: str
    page: int
    score: float
    text: str


@dataclass
class CitationRegistry:
    """Gives every retrieved chunk a short, stable id (S1, S2 ...) for this
    chat turn. The same chunk found by two searches keeps the same id."""
    sources: dict[str, Source] = field(default_factory=dict)
    _by_chunk: dict[tuple[str, int], str] = field(default_factory=dict)

    def register(self, hit: dict) -> str:
        key = (hit["doc_id"], hit["chunk_index"])
        if key not in self._by_chunk:
            sid = f"S{len(self._by_chunk) + 1}"
            self._by_chunk[key] = sid
            self.sources[sid] = Source(sid, hit["doc_id"], hit["filename"], hit["page"],
                                       hit["score"], hit["text"])
        return self._by_chunk[key]


@dataclass
class ToolContext:
    knowledge: KnowledgeClient
    citations: CitationRegistry
    default_top_k: int                  # from the UI settings (or the server default)
    max_chars: int
    score_threshold: float | None = None
    embedding_model: str | None = None


# ---------------------------------------------------------------------------- arguments
class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500, description=(
        "A focused search query. Rewrite the user's question into the key terms "
        "likely to appear in the document, e.g. 'TIV referral threshold chief underwriter'."))
    top_k: int = Field(default=0, ge=0, le=10,
                       description="How many passages to return; 0 = use the user's setting")


class ListDocumentsArgs(BaseModel):
    pass


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200, description=(
        "Arithmetic expression using numbers, + - * / // % ** and parentheses, "
        "e.g. '(250000000 * 0.15) / 12'"))


# ---------------------------------------------------------------------------- handlers
async def search_knowledge_base(args: SearchArgs, ctx: ToolContext) -> dict:
    try:
        hits = await ctx.knowledge.search(args.query, args.top_k or ctx.default_top_k,
                                          score_threshold=ctx.score_threshold,
                                          embedding_model=ctx.embedding_model)
    except KnowledgeUnavailable as exc:
        return {"error": f"The knowledge base is temporarily unavailable: {exc}. "
                         "Tell the user you cannot search the documents right now."}
    if not hits:
        return {"results": [], "note": "No relevant passages found. Try different keywords, "
                                       "or tell the user the documents do not cover this."}
    results, used = [], 0
    for h in hits:
        sid = ctx.citations.register(h)
        text = h["text"]
        if used + len(text) > ctx.max_chars:            # keep the prompt within budget
            text = text[: max(0, ctx.max_chars - used)]
        used += len(text)
        results.append({"source_id": sid, "filename": h["filename"], "page": h["page"],
                        "score": h["score"], "text": text})
        if used >= ctx.max_chars:
            break
    return {"results": results, "note": "Cite facts with the source_id in square brackets."}


async def list_documents(_: ListDocumentsArgs, ctx: ToolContext) -> dict:
    try:
        docs = await ctx.knowledge.list_documents(ctx.embedding_model)
    except KnowledgeUnavailable as exc:
        return {"error": f"The knowledge base is temporarily unavailable: {exc}"}
    return {"documents": [{"filename": d["filename"], "pages": d.get("pages"),
                           "chunks": d.get("total_chunks"), "uploaded_at": d.get("uploaded_at")}
                          for d in docs]}


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


async def calculator(args: CalculatorArgs, _: ToolContext) -> dict:
    try:
        return {"expression": args.expression, "result": safe_eval(args.expression)}
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return {"error": f"Cannot evaluate '{args.expression}': {exc}"}


# ---------------------------------------------------------------------------- registry
@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Awaitable[dict]]

    def schema(self) -> dict:
        params = _strip_titles(self.args_model.model_json_schema())
        params.setdefault("properties", {})
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": params}}


def _strip_titles(schema):
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_knowledge_base",
         "Semantic search over the user's uploaded documents. Returns the most relevant "
         "passages with a source_id to cite.", SearchArgs, search_knowledge_base),
    Tool("list_documents", "List the documents currently in the knowledge base.",
         ListDocumentsArgs, list_documents),
    Tool("calculator", "Evaluate an arithmetic expression exactly.", CalculatorArgs, calculator),
]}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


async def execute_tool(name: str, raw_args: dict, ctx: ToolContext) -> tuple[dict, str]:
    """Run one tool call. Returns (result, status). Never raises: every
    failure becomes a message the model can read and react to."""
    tool = TOOLS.get(name)
    start = time.perf_counter()
    if tool is None:
        result, status = {"error": f"Unknown tool '{name}'. Available: {list(TOOLS)}"}, "unknown"
    else:
        try:
            args = tool.args_model.model_validate(raw_args)
            result = await tool.handler(args, ctx)
            status = "error" if "error" in result else "ok"
        except ValidationError as exc:
            result, status = {"error": f"Invalid arguments: {exc.errors(include_url=False)}"}, "invalid"
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the chat turn
            result, status = {"error": f"Tool '{name}' failed: {type(exc).__name__}: {exc}"}, "error"
    m.TOOL_CALLS.labels(name if tool else "unknown", status).inc()
    m.TOOL_LATENCY.labels(name if tool else "unknown").observe(time.perf_counter() - start)
    return result, status


def to_tool_message(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
