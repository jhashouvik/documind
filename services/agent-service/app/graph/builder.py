"""Wire the nodes into the top-level StateGraph.

Static edges are drawn here. Dynamic routing (supervisor, planner, grounding
check) happens inside the nodes by returning Command(goto=...); their return
type annotations (Command[Literal[...]]) tell LangGraph the possible targets,
so the graph can still be drawn:  GET /v1/graph  returns it as Mermaid.

compile() turns the builder into a runnable Pregel graph and attaches:
  checkpointer  short-term memory: state of every thread after every step
                (resume after an interrupt, time travel, fault tolerance)
  store         long-term memory: key-value documents shared across threads
"""
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from ..models import is_transient
from .context import AgentContext
from .nodes import (ResearchTask, analyst, finalize, grounding_check, librarian, planner, remember,
                    research_task, researcher, start_turn, supervisor, writer)
from .state import DocuMindState

AGENTS = ["supervisor", "researcher", "planner", "analyst", "librarian", "writer", "grounding_check"]


def build_graph() -> StateGraph:
    retry = RetryPolicy(max_attempts=2, initial_interval=1.0, retry_on=is_transient)
    g = StateGraph(DocuMindState, context_schema=AgentContext)
    g.add_node("start_turn", start_turn)
    g.add_node("supervisor", supervisor, retry_policy=retry)
    g.add_node("researcher", researcher)
    g.add_node("planner", planner, retry_policy=retry)
    g.add_node("research_task", research_task, input_schema=ResearchTask)
    g.add_node("analyst", analyst)
    g.add_node("librarian", librarian)
    g.add_node("writer", writer)
    g.add_node("grounding_check", grounding_check, retry_policy=retry)
    g.add_node("finalize", finalize)
    g.add_node("remember", remember)

    g.add_edge(START, "start_turn")
    g.add_edge("start_turn", "supervisor")
    for worker in ("researcher", "research_task", "analyst", "librarian"):
        g.add_edge(worker, "supervisor")          # every worker reports back to the supervisor
    g.add_edge("writer", "grounding_check")
    g.add_edge("finalize", "remember")
    g.add_edge("remember", END)
    return g


def compile_graph(checkpointer=None, store=None):
    return build_graph().compile(checkpointer=checkpointer, store=store, name="documind")
