"""The researcher: a corrective-RAG (CRAG) subgraph.

Plain RAG trusts whatever the vector search returns. Corrective RAG CHECKS:

    START -> retrieve -> grade --(relevant passages found)--> END
                ^          |
                |          +--(nothing relevant, rewrites left)--> rewrite
                +------------------------------------------------------+

  retrieve  search knowledge-service with the current query
  grade     an LLM judge keeps only passages that help answer the question
  rewrite   if nothing helped, an LLM writes a better query and we search again

It is its own StateGraph with its own state (ResearchState). The parent graph
runs it from the `researcher` node (one question) and from `research_task`
(many questions in parallel, see planner in nodes.py).
"""
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .. import metrics as m
from ..knowledge_client import KnowledgeUnavailable, SearchHit
from .common import decide, emit
from .context import AgentContext
from .prompts import GRADER, REWRITER
from .state import Grades, Passage, ResearchState, Rewrite


def _passage(hit: SearchHit, query: str) -> Passage:
    return Passage(key=f"{hit.doc_id}:{hit.chunk_index}", query=query,
                   **hit.model_dump(include={"doc_id", "filename", "page", "chunk_index", "score", "text"}))


async def retrieve(state: ResearchState, runtime: Runtime[AgentContext]) -> dict:
    ctx, cfg = runtime.context, runtime.context.cfg
    query = state.get("query") or state["question"]
    tried = [*(state.get("tried") or []), query]
    try:
        hits = await ctx.knowledge.search(query, cfg.top_k, score_threshold=cfg.score_threshold,
                                          embedding_model=cfg.embedding_model)
    except KnowledgeUnavailable as exc:
        emit("researcher", "search", f'"{query}": {exc}', "error")
        return {"candidates": [], "tried": tried, "query": query, "error": str(exc)}
    files = sorted({h.filename for h in hits})
    emit("researcher", "search", f'"{query}" -> {len(hits)} passages'
                                 + (f" from {', '.join(files)}" if files else ""))
    return {"candidates": [_passage(h, query) for h in hits], "tried": tried, "query": query}


async def grade(state: ResearchState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    cands = state.get("candidates") or []
    if not cands:
        return {}
    listing = "\n\n".join(f"[{i}] ({p['filename']}, page {p['page']}) {p['text'][:1500]}"
                          for i, p in enumerate(cands, 1))
    grades = await decide(ctx.models.structured(ctx.cfg, "grader", Grades),
                          [SystemMessage(GRADER),
                           HumanMessage(f"Question: {state['question']}\n\nPassages:\n{listing}")],
                          Grades, Grades(relevant=list(range(1, len(cands) + 1))),   # fail open: keep all
                          context={"n_passages": len(cands)})
    keep = [cands[i - 1] for i in sorted(set(grades.relevant)) if 1 <= i <= len(cands)]
    seen = {p["key"] for p in state.get("relevant") or []}
    relevant = [*(state.get("relevant") or []), *(p for p in keep if p["key"] not in seen)]
    emit("researcher", "grade", f"{len(keep)} of {len(cands)} passages relevant",
         "ok" if keep else "error")
    return {"relevant": relevant}


def after_grade(state: ResearchState, runtime: Runtime[AgentContext]) -> Literal["rewrite", "__end__"]:
    if state.get("error") or state.get("relevant"):
        return END
    if (state.get("attempts") or 0) < runtime.context.settings.max_query_rewrites:
        return "rewrite"
    return END


async def rewrite(state: ResearchState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    tried = "\n".join(f"- {q}" for q in state.get("tried") or [])
    new = await decide(ctx.models.structured(ctx.cfg, "rewriter", Rewrite),
                       [SystemMessage(REWRITER),
                        HumanMessage(f"Question: {state['question']}\n\nQueries already tried:\n{tried}")],
                       Rewrite, None, context={"tried": state.get("tried") or []})
    if new is None or not new.query.strip() or new.query in (state.get("tried") or []):
        emit("researcher", "rewrite", "no better query", "error")
        return {"attempts": (state.get("attempts") or 0) + 1, "query": state.get("query")}
    m.QUERY_REWRITES.inc()
    emit("researcher", "rewrite", f'new query "{new.query}"')
    return {"query": new.query, "attempts": (state.get("attempts") or 0) + 1}


def after_rewrite(state: ResearchState) -> Literal["retrieve", "__end__"]:
    tried = state.get("tried") or []
    return END if tried and state.get("query") == tried[-1] else "retrieve"


def build_research_graph():
    g = StateGraph(ResearchState, context_schema=AgentContext)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade)
    g.add_conditional_edges("rewrite", after_rewrite)
    return g.compile(name="researcher")


RESEARCH_GRAPH = build_research_graph()
